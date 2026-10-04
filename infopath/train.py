import torch
import numpy as np
import os
import time
from optparse import OptionParser
from datasets.dataloader import balance_trial_type, keep_trial_types_behaviour
from infopath.model_loader import load_model_and_optimizer
from infopath.config import get_opt, save_opt, config_vahid, config_pseudodata
from infopath.utils.functions import (
    load_data,
    prepare_svd_feat,
    return_trial_type,
    trial_metric,
    trial_type_perc,
)
from infopath.utils.logger import get_log_path, init_results, log_model_and_results_gan, optimizer_to
from infopath.utils.plot_utils import plot_rsnn_activity
from infopath.lick_classifier import prepare_classifier
from infopath.losses import discriminator_loss, find_important_elements
from infopath.utils.online_logger import OnlineLogger
from infopath.utils.diagnostics import DiagnosticsLogger, count_trainable_parameters, Timer

# import wandb


def train(opt, model, netD, optimizerG, optimizerD, step=-1):
    if opt.verbose:
        print(model)
    results = init_results(opt.log_path)
    early_stop, accuracy, previous_test_loss, t_trial_pearson_model = (
        0,
        torch.zeros(1),
        1000,
        0,
    )
    life = 0
    to_save_lists = {
        "total_train_loss": [],
        "neuron_loss": [],
        "trial_loss": [],
        "fr_loss": [],
        "cross_corr_loss": [],
        "t_trial_pearson": [],
        "tm_mle_loss": [],
        "data_loss": [],
        "dt_forward": [],
        "dt_backward": [],
        "dt_step": [],
    }

    # ---- diagnostics logger: log JSONL con todo lo que se pueda para
    # inspeccionar el training (actividad de neuronas, pesos activos, timing) ----
    diag_logger = DiagnosticsLogger(opt.log_path, step_every=50, flush_every=1)
    count_trainable_parameters(model, verbose=True)
    online = OnlineLogger(opt, name=os.path.basename(opt.log_path))
    diag_logger.log(
        step=step,
        model=model,
        log_params=True,
        extra={"event": "train_start"},
    )

    if opt.with_behaviour:
        raise NotImplementedError(
            "with_behaviour=True is not supported for the Pierre data yet: lick_classifier.prepare_classifier "
            "needs lick_counts_train/test (never computed), pierre/process.py writes no whisker/tongue traces, "
            "and the lick traces are 1 kHz while dataloader.py assumes 500 Hz."
        )
    if opt.iterative_pruning:
        init_pruning(model)
    # load data and prepare lick_classifier
    (
        train_spikes,
        train_jaw,
        session_info_train,
        test_spikes,
        test_jaw,
        session_info_test,
    ) = load_data(model)
    if opt.feat_svd:
        filt = model.filter_fun2(model.filter_fun1(train_spikes))
        prepare_svd_feat(model, filt)
    stim_cond = [opt.trial_types]
    log_step = opt.log_every_n_steps - (opt.log_every_n_steps % len(stim_cond))
    if getattr(opt, "resume", False) and results["test_loss"]:
        step = len(results["test_loss"]) * log_step - 1
        previous_test_loss = min(results["test_loss"])
        print(f"resuming at step {step + 1} (previous best test loss {previous_test_loss:.4f})")

    data_spikes = train_spikes.clone()
    data_jaw = None
    filt_jaw_test = None
    filt_jaw_train = None
    lick_classifier = None
    if opt.with_behaviour:
        filt_jaw_train = model.filter_fun1(train_jaw)
        filt_jaw_test = model.filter_fun1(test_jaw)
        data_jaw = train_jaw.clone()
        lick_classifier = prepare_classifier(
            model.filter_fun2(filt_jaw_train),
            model.filter_fun2(filt_jaw_test),
            session_info_train,
            session_info_test,
            opt.device,
        )
    t_trial_pearson_data = t_trial_pearson(
        model.filter_fun1(train_spikes),
        model.filter_fun1(test_spikes),
        filt_jaw_train,
        filt_jaw_test,
        model,
        session_info_train,
    )
    data_perc = trial_type_perc(session_info_train)
    if opt.verbose:
        print("T'_{trial} data train and test", t_trial_pearson_data)
        print("the data have trial type distribution: ", data_perc)

    optimizerG.param_groups[0]["initial_lr"] = optimizerG.param_groups[0]["lr"]
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.99)
    if opt.gan_loss:
        optimizerD.param_groups[0]["initial_lr"] = optimizerD.param_groups[0]["lr"]
        schedulerD = torch.optim.lr_scheduler.StepLR(
            optimizerD, step_size=100, gamma=0.99
        )

    if opt.loss_cross_corr:
        try:
            data_ccg = np.load(os.path.join(opt.datapath, "ccg.npy"))
            model.cc_data = torch.tensor(data_ccg, device=opt.device)
        except FileNotFoundError:
            find_important_elements(model, train_spikes, session_info_train)

    mem_loss = 0
    while step < opt.n_steps:
        step += 1
        # prepare data for stimulus condition, loop with all, only stim, only no stim
        data_spikes = train_spikes
        data_jaw = train_jaw
        session_info = session_info_train
        all_stims = np.concatenate([stims for stims in session_info[1]])
        stims = all_stims[torch.randint(all_stims.shape[0], size=(opt.batch_size,))]
        stims = torch.tensor(stims, device=opt.device)
        thr_rest = (opt.thr - opt.v_rest) / 0.1
        # Train Discriminator
        if step % 4 == 0 and opt.gan_loss:
            optimizerD.zero_grad()
            model.zero_grad()
            model_spikes, voltages, model_jaw, state = model(stims, step)
            if opt.t_trial_gan:
                model_spikes_gan = model.filter_fun2(model.filter_fun1(model_spikes))
                model_jaw_gan = model.filter_fun2(model.filter_fun1(model_jaw))
                data_spikes_gan = model.filter_fun2(model.filter_fun1(data_spikes))
                data_jaw_gan = model.filter_fun2(model.filter_fun1(data_jaw))
            else:
                model_spikes_gan = model_spikes
                model_jaw_gan = model_jaw
                data_spikes_gan = data_spikes
                data_jaw_gan = data_jaw

            lossD, accuracy = discriminator_loss(
                netD,
                model_spikes_gan,
                data_spikes_gan,
                data_jaw_gan,
                model_jaw_gan,
                session_info,
                model.rsnn.area_index,
                model.rsnn.excitatory_index,
                discriminator=True,
                t_trial_gan=opt.t_trial_gan,
                z_score=opt.z_score,
            )
            lossD.backward()
            optimizerD.step()
            # schedulerD.step()

        ## Train generator, the last condition is for go_back_to_hunting
        if not opt.gan_loss or step % 1 == 0 or step % log_step == 1:
            # Train Generator]
            grad_norm = None
            with Timer(opt.device) as t_fwd:
                optimizerG.zero_grad()
                model.zero_grad()
                model_spikes, volt, model_jaw, state = model(stims, step)
                if model_spikes.min().isnan():
                    diag_logger.log(
                        step=step, extra={"event": "explosion_reset", "life": life}
                    )
                    model, netD, optimizerG, optimizerD, life, step = explosion_reset(
                        opt, life, step
                    )
                    to_save_lists = {i: [] for i in to_save_lists.keys()}
                    continue
                (
                    fr_loss,
                    trial_loss,
                    neuron_loss,
                    cross_corr_loss,
                    tm_mle_loss,
                ) = model.generator_loss(
                    model_spikes,
                    data_spikes,
                    model_jaw,
                    data_jaw,
                    session_info,
                    netD,
                    stims=stims,
                )

                if opt.loss_mem_volt:
                    mem_loss = (
                        ((volt - model.rsnn.v_rest) / thr_rest) ** 2
                    ).mean() ** 0.5

                conn_loss = 0
                across = model.rsnn.off_diag.to(opt.device)
                # for teacher only to set the shape connection
                if "2areas_abstract" in opt.datapath and opt.block_graph == []:
                    area1 = model.rsnn.area_index == 0
                    area2 = model.rsnn.area_index == 1
                    connection_strength_going = across.clone()
                    connection_strength_going[:, area2] = 0
                    connection_strength_coming = across.clone()
                    connection_strength_coming[:, area1] = 0
                    conn_going = model.rsnn._w_rec[:, connection_strength_going].sum()
                    conn_coming = model.rsnn._w_rec[:, connection_strength_coming].sum()
                    conn_loss = 0.1 * (
                        conn_going / opt.diff_strength - conn_coming
                    ).abs()

                total_train_loss = (
                    neuron_loss
                    + trial_loss
                    + fr_loss
                    + cross_corr_loss
                    + tm_mle_loss
                    + mem_loss
                    + conn_loss
                )
            dt_forward = t_fwd.dt

            with Timer(opt.device) as t_bwd:
                total_train_loss.backward()
            dt_backward = t_bwd.dt

            if opt.clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt.clip_grad
                )

            with Timer(opt.device) as t_step:
                w_rec = model.rsnn._w_rec.data.clone()
                if opt.with_behaviour:
                    w_jaw = model.rsnn._w_jaw_pre.data.clone()
                else:
                    w_jaw = 0
                w_in = model.rsnn._w_in.data.clone()
                optimizerG.step()

                with torch.no_grad():
                    weight_decay(model, w_rec, w_jaw, w_in)
                    if opt.flag_ei:
                        model.rsnn.reform_recurent(opt.lr, l1_decay=0)
                    if not opt.flag_ei and opt.restrict_inter_area_inh:
                        model.rsnn._w_rec.data *= model.rsnn.mask_inter_inh_exc
                    model.rsnn.reform_v_rest()
            dt_step = t_step.dt

            if (step % log_step == 1) & opt.iterative_pruning:
                iterative_pruning(model)

            state = [[st.detach() for st in state[0]]]

        if opt.verbose:
            print_loss(
                opt,
                total_train_loss,
                neuron_loss,
                trial_loss,
                fr_loss,
                cross_corr_loss,
                accuracy,
                conn_loss,
                step,
            )

            if step % (10 - (10 % len(stim_cond))) == 0:
                with torch.no_grad():
                    t_trial_pearson_model = t_trial_pearson(
                        model.filter_fun1(data_spikes),
                        model.filter_fun1(model_spikes),
                        model.filter_fun1(data_jaw),
                        model.filter_fun1(model_jaw),
                        model,
                        session_info,
                    )
                print("t_trial_pearson: ", t_trial_pearson_model)

        to_save_lists["total_train_loss"].append(float(total_train_loss))
        to_save_lists["neuron_loss"].append(float(neuron_loss))
        to_save_lists["trial_loss"].append(float(trial_loss))
        to_save_lists["fr_loss"].append(float(fr_loss))
        to_save_lists["cross_corr_loss"].append(float(cross_corr_loss))
        to_save_lists["tm_mle_loss"].append(float(tm_mle_loss))
        to_save_lists["data_loss"].append(float(total_train_loss - conn_loss))
        to_save_lists["dt_forward"].append(dt_forward)
        to_save_lists["dt_backward"].append(dt_backward)
        to_save_lists["dt_step"].append(dt_step)
        if t_trial_pearson_model != 0:
            to_save_lists["t_trial_pearson"].append(float(t_trial_pearson_model))

        # ---- diagnostics: log ligero cada step, stats caras (actividad de
        # neuronas / pesos activos) cada `step_every` steps (ver DiagnosticsLogger) ----
        diag_logger.log(
            step=step,
            model=model,
            model_spikes=model_spikes,
            extra={
                "epoch": step // log_step,
                "total_train_loss": float(total_train_loss),
                "neuron_loss": float(neuron_loss),
                "trial_loss": float(trial_loss),
                "fr_loss": float(fr_loss),
                "cross_corr_loss": float(cross_corr_loss),
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "lr": optimizerG.param_groups[0]["lr"],
                "dt_forward": dt_forward,
                "dt_backward": dt_backward,
                "dt_step": dt_step,
            },
        )

        online.log(
            {
                "total_train_loss": float(total_train_loss),
                "neuron_loss": float(neuron_loss),
                "trial_loss": float(trial_loss),
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "dt_forward": dt_forward,
                "dt_backward": dt_backward,
                "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            },
            step,
        )
        if step % log_step == 0 and torch.cuda.is_available():
            print(f"gpu peak allocated so far: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

        if step % log_step == 0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                bs = opt.batch_size
                if "Vahid" in opt.datapath:
                    opt.batch_size = opt.batch_size
                else:
                    opt.batch_size = getattr(opt, "eval_batch_size", 400)
                # torch.manual_seed(0)
                stims = all_stims[
                    torch.randint(all_stims.shape[0], size=(opt.batch_size,))
                ]
                stims = torch.tensor(stims, device=opt.device)
                model.eval()
                state = model.steady_state()
                input_spikes = model.input_spikes(stims)
                model.rsnn.sample_mem_noise(
                    input_spikes.shape[0], input_spikes.shape[1]
                )
                mem_noise = model.rsnn.mem_noise.clone()
                model_spikes, voltages, model_jaw, state = model.step(
                    input_spikes, state, mem_noise=mem_noise
                )
                stats_loss = opt.stats_loss
                opt.stats_loss = False
                activity_figure = prepare_plot(
                    model,
                    model_spikes,
                    voltages,
                    model_jaw,
                    input_spikes,
                    opt,
                    test_spikes,
                    test_jaw,
                    session_info_test,
                    lick_classifier,
                    stims,
                )
                (
                    test_loss,
                    trial_type_acc,
                    t_trial_pearson_ratio,
                    eval_trial_type,
                    eval_model_perc,
                ) = goodness_of_fit(
                    test_spikes,
                    model_spikes,
                    test_jaw,
                    model_jaw,
                    session_info_test,
                    stims,
                    model,
                    data_perc,
                    t_trial_pearson_data,
                    lick_classifier,
                    netD,
                )
                opt.batch_size = bs
                opt.stats_loss = stats_loss
                if opt.verbose:
                    print(
                        f"trial type accuracy {trial_type_acc} ratio of t_trial_test/t_trial_data {t_trial_pearson_ratio}"
                    )

            online.log_eval(
                step, model, model_spikes, test_spikes, stims,
                eval_trial_type, eval_model_perc, data_perc,
            )
            results["trial_type_accuracy"].append(trial_type_acc)
            results["t_trial_pearson_ratio"].append(t_trial_pearson_ratio.item())
            for key in to_save_lists.keys():
                results[key].append(np.mean(to_save_lists[key]))
            results["test_loss"].append(test_loss)
            online.log(
                {
                    "test_loss": float(test_loss),
                    "trial_type_accuracy": float(trial_type_acc),
                    "t_trial_pearson_ratio": float(t_trial_pearson_ratio.item()),
                },
                step,
            )

            # ---- diagnostics: entrada de eval, usa el batch de test para
            # stats de actividad neuronal completas (force_full=True: el eval
            # solo corre cada log_step steps, así que es barato hacerlo siempre) ----
            diag_logger.log(
                step=step,
                model=model,
                model_spikes=model_spikes,
                force_full=True,
                extra={
                    "event": "eval",
                    "test_loss": float(test_loss),
                    "trial_type_accuracy": float(trial_type_acc),
                    "t_trial_pearson_ratio": float(t_trial_pearson_ratio.item()),
                    "early_stop_counter": early_stop,
                    "mean_dt_forward_epoch": float(results["dt_forward"][-1]),
                    "mean_dt_backward_epoch": float(results["dt_backward"][-1]),
                    "mean_dt_step_epoch": float(results["dt_step"][-1]),
                },
            )

            best_loss = "data_loss" if "Vahid" in opt.datapath else "test_loss"
            log_model_and_results_gan(
                opt,
                model,
                optimizerG,
                results,
                results[best_loss][-1],
                step,
                [activity_figure],
                netD=netD,
                optimizerD=optimizerD,
            )
            model.train()
            for key in to_save_lists.keys():
                to_save_lists[key] = []
            if results[best_loss][-1] > previous_test_loss:
                if early_stop > opt.early_stop:
                    print("probably pointless to continue so I stop myself")
                    print("So long, and thanks for the fish")
                    diag_logger.close()
                    return -1
            else:
                previous_test_loss = results[best_loss][-1]
                early_stop = 0
            torch.cuda.empty_cache()
            entry = {}
            for i in results.keys():
                if (
                    results[i] != []
                    and results[i] is not None
                    and type(results[i]) == list
                ):
                    entry[i] = results[i][-1]

            # wandb.log(entry)
        early_stop += 1
    print("So long, and thanks for the fish")
    diag_logger.close()
    return -1


def explosion_reset(opt, life, step):
    step -= step % opt.log_every_n_steps
    model, netD, optG, optD = load_model_and_optimizer(opt, reload=True)
    life += 1
    if life > 3:
        print("I exploded 3 times, I give up")
        return -1
    return model, netD, optG, optD, life, step


def t_trial_pearson(
    filt_data, filt_model, data_jaw, model_jaw, model, session_info, measure="pear_corr"
):
    with torch.no_grad():
        t_trial_p = trial_metric(
            model.filter_fun2(filt_data),
            model.filter_fun2(filt_model),
            model.filter_fun2(data_jaw),
            model.filter_fun2(model_jaw),
            session_info,
            model,
            measure,
        )
        return t_trial_p.mean()


@torch.no_grad()
def weight_decay(model, w_rec, w_jaw, w_in):
    opt = model.opt
    across = model.rsnn.off_diag.to(opt.device)
    valid = w_rec > 1e-7
    model.rsnn._w_rec.data[valid] -= opt.l1_decay * opt.lr / w_rec[valid] ** 0.5
    model.rsnn._w_rec.data[valid & across] -= (
        opt.l1_decay_across * opt.lr / w_rec[valid & across] ** 0.5
    )
    if model.rsnn._w_in.requires_grad:
        valid = w_in > 1e-7
        model.rsnn._w_in.data[valid] -= (
            opt.l1_decay_across * opt.lr / w_in[valid] ** 0.5
        )
    if model.opt.with_behaviour:
        # actual l1
        model.rsnn._w_jaw_pre.data -= opt.l1_decay_across * opt.lr * w_jaw.abs() * 10


def init_pruning(model):
    for param in ["_w_in", "_w_rec", "_w_jaw_pre"]:
        if hasattr(model.rsnn, param):
            if getattr(model.rsnn, param).requires_grad:
                mask = getattr(model.rsnn, "mask_prune" + param)
                mask = mask.to(model.opt.device)
                mask[getattr(model.rsnn, param).abs() < 1e-7] = 1
                setattr(model.rsnn, "mask_prune" + param, mask)


def iterative_pruning(model, perc=0.2, perc_off=0.1, stop_at=0.1):
    ### Prune the 20% smallest of the remaining weights of the parameters: _w_in, _w_rec and _w_jaw_pre
    parameters = []
    for param in ["_w_in", "_w_rec", "_w_jaw_pre"]:
        if hasattr(model.rsnn, param):
            if getattr(model.rsnn, param).requires_grad:
                parameters.append(param)
    for param in parameters:
        # mask
        mask = getattr(model.rsnn, "mask_prune" + param).to(model.opt.device)
        # snapshot
        # previous_active = f"active{1000*(1 - mask.float().mean()):.0f}"
        # model_path = os.path.join(model.opt.log_path, f"{previous_active}_model.ckpt")
        # torch.save(model.state_dict(), model_path)
        parameter = getattr(model.rsnn, param).data
        ones = mask.sum().item()
        quantile = int(ones + perc * (parameter.numel() - ones)) / parameter.numel()
        if param == "_w_rec":
            on_diag = ~model.rsnn.off_diag
            on_diag_sum = on_diag.sum().item()
            on_diag = on_diag.to(mask.device) & mask.bool()
            if 1 - on_diag.float().sum() / on_diag_sum < stop_at:
                print("No more general pruning")
                quantile = 0
        mask[
            torch.abs(parameter)
            < torch.quantile(torch.abs(parameter).flatten(), quantile)
        ] = 1
        if param == "_w_rec":
            value = parameter[:, model.rsnn.off_diag]
            value = value[value > 0]
            quantile = torch.quantile(value, perc_off)
            off_mask = torch.abs(parameter) < quantile
            off_mask = off_mask & model.rsnn.off_diag.to(off_mask.device)
            mask = mask.bool() + off_mask.bool()

        setattr(model.rsnn, "mask_prune" + param, mask)
        print(f"active perc: {1 - mask.float().mean():.3f}")


def prepare_plot(
    model,
    model_spikes,
    voltages,
    model_jaw,
    input_spikes,
    opt,
    data_spikes,
    data_jaw,
    session_info,
    lick_classifier=None,
    stims=None,
):
    filt_model = model.filter_fun1(model_spikes)
    filt_data = model.filter_fun1(data_spikes)
    filt_model_jaw = model.filter_fun1(model_jaw)
    filt_data_jaw = model.filter_fun1(data_jaw)
    # statistics of trial variability
    trial_type, model_perc = return_trial_type(
        model,
        filt_data,
        filt_model,
        filt_data_jaw,
        filt_model_jaw,
        session_info,
        stims,
        lick_classifier,
    )
    if opt.verbose:
        print("trial type distribution for the model", model_perc)

    # plot the mean neural activity per trial type and area
    mean_signals = []
    for tr_type in opt.trial_types:
        data_spikes_trty = test_hit_miss(data_spikes, session_info, trial_type=tr_type)
        exc, _ = model.mean_activity(data_spikes_trty)
        mean_signals.append(exc)

    for tr_type in opt.trial_types:
        if (trial_type == tr_type).sum() == 0:
            # plot zeros if the model predict nothing for that trial type
            signal = [np.zeros_like(sig) for sig in mean_signals[0]]
        else:
            signal, _ = model.mean_activity(model_spikes[:, trial_type == tr_type])
        mean_signals.append(signal)

    firing_rates_t = data_spikes[: model.trial_onset].nanmean((0, 1)) / model.timestep
    # calculating the baseline firing rate
    firing_rates = model_spikes[: model.trial_onset].mean(0).mean(0) / model.timestep
    firing_rates = firing_rates[model.neuron_index != -1]
    firing_rates_t = firing_rates_t[model.neuron_index != -1]
    activity_figure, _ = plot_rsnn_activity(
        input=input_spikes[:, -1, :].cpu().numpy(),
        spikes=model_spikes[:, -1, :].cpu().numpy(),
        voltages=voltages[:, -1, :].cpu().numpy(),
        output=mean_signals,
        jaw=[model_jaw, model_jaw],
        firing_rates=firing_rates.cpu().detach().numpy(),
        firing_rates_target=firing_rates_t.cpu().detach().numpy(),
        n_neuron_max=500,
        areas=opt.num_areas,
        dt=(opt.stop - opt.start) / mean_signals[0][0].shape[0],
        spike_function=opt.spike_function,
    )

    return activity_figure


def goodness_of_fit(
    data_spikes,
    model_spikes,
    data_jaw,
    model_jaw,
    session_info,
    stims,
    model,
    data_perc,
    t_trial_pearson_data,
    lick_classifier=None,
    netD=None,
):
    opt = model.opt
    (
        fr_loss,
        trial_loss,
        neuron_loss,
        cross_corr_loss,
        tm_mle_loss,
    ) = model.generator_loss(
        model_spikes, data_spikes, model_jaw, data_jaw, session_info, netD, stims=stims
    )
    filt_model = model.filter_fun1(model_spikes)
    filt_data = model.filter_fun1(data_spikes)
    filt_model_jaw = model.filter_fun1(model_jaw)
    filt_data_jaw = model.filter_fun1(data_jaw)
    test_loss = (
        neuron_loss + trial_loss + fr_loss + cross_corr_loss + tm_mle_loss
    ).item()
    print_loss(
        opt,
        test_loss,
        neuron_loss,
        trial_loss,
        fr_loss,
        cross_corr_loss,
        None,
        0,
        None,
    )

    trial_type, model_perc = return_trial_type(
        model,
        filt_data,
        filt_model,
        filt_data_jaw,
        filt_model_jaw,
        session_info,
        stims,
        lick_classifier,
    )

    data_perc_95 = 1.96 * (data_perc * (1 - data_perc) / opt.batch_size) ** 0.5
    acc_trial_type = (np.abs((model_perc.numpy() - data_perc)) < data_perc_95).sum()
    acc_trial_type /= len(opt.trial_types)
    t_trial_pearson_test = t_trial_pearson(
        filt_data, filt_model, filt_data_jaw, filt_model_jaw, model, session_info
    )
    t_trial_pearson_ratio = t_trial_pearson_test / t_trial_pearson_data
    print("t_trial_pearson: ", t_trial_pearson_test)

    return test_loss, acc_trial_type, t_trial_pearson_ratio, trial_type, model_perc


def test_hit_miss(data_spikes, session_info, trial_type=1):
    T, trials, neurons = data_spikes.shape
    max_trials = max([(i == trial_type).sum() for i in session_info[0]])
    test_data = torch.ones(T, max_trials, neurons) * torch.nan
    for sess in range(len(session_info[0])):
        trial_types = np.where(session_info[0][sess] == trial_type)[0]
        idx = session_info[-1][sess]
        sess_data = data_spikes[..., idx][:, trial_types].cpu()
        resid_trials = max_trials - trial_types.shape[0]
        sess_data = torch.cat(
            [sess_data, torch.ones(T, resid_trials, idx.sum()) * torch.nan], dim=1
        )
        test_data[:, :, session_info[-1][sess]] = sess_data
    return test_data


def print_loss(
    opt,
    total_train_loss,
    neuron_loss,
    trial_loss,
    fr_loss,
    cross_corr_loss,
    accuracy,
    l05_loss,
    step,
):
    loss_str = f"total_loss {total_train_loss:.2f} "
    if opt.loss_neuron_wise:
        loss_str += f" | neuron_loss {neuron_loss:.2f}"
    if opt.loss_firing_rate:
        loss_str += f" | fr loss {fr_loss:.2f}"
    if opt.loss_trial_wise:
        loss_str += f" | trial_loss {trial_loss:.2f}"
    if opt.loss_cross_corr:
        loss_str += f" | cross_corr {cross_corr_loss:.4f}"
    if opt.gan_loss:
        if accuracy is not None:
            loss_str += f" | discriminator {accuracy.item():.2f}"
    if l05_loss > 0:
        loss_str += f" | l05 loss {l05_loss:.2f}"
    print("step", step, " ", loss_str)


def git_diff(log_path):
    stream = os.popen("git diff infopath models")
    output = stream.read()
    with open(log_path + "/diff.txt", "w") as f:
        f.writelines(output)


if __name__ == "__main__":
    # here either you set the config from a file that you create with >> python3 infopath/config.py --config=$NAME
    # if there is no arg in the parser then is what is currently the default in the infopath/config.py
    parser = OptionParser()
    parser.add_option("--config", type="string", default="none")
    parser.add_option("--resume", type="string", default=None,
                      help="log_path of a previous run: reload last_model/last_optim and continue")
    (pars, _) = parser.parse_args()

    if pars.config == "none":
        opt = config_pseudodata()
        config = "daniel"
        opt = get_opt(os.path.join("configs", config))
        if "/" in config:
            config = config.split("/")[-1]
        get_log_path(opt, config)
    else:
        opt = get_opt(os.path.join("configs", pars.config))
        if pars.resume:
            opt.log_path = os.path.abspath(pars.resume)
            opt.resume = True
        config = pars.config
        if "/" in config:
            config = config.split("/")[-1]
        print(config)
        get_log_path(opt, config)
    import warnings

    warnings.filterwarnings("ignore")
    if opt.verbose:
        print("log_path", opt.log_path)

    git_diff(opt.log_path)

    # set random seeds
    if opt.seed >= 0:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)

    # load model
    dev = (
        opt.device
        if opt.device.type == "cuda" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    opt.device = torch.device("cpu")
    model, netD, optimizerG, optimizerD = load_model_and_optimizer(
        opt, reload=getattr(opt, "resume", False)
    )
    opt.device = dev
    model.to(opt.device)
    # optimizer state loaded on resume was placed on the CPU (opt.device was cpu during loading)
    optimizer_to(optimizerG, opt.device)
    if optimizerD is not None:
        optimizer_to(optimizerD, opt.device)
    if netD is not None:
        netD.to(opt.device)
    # wandb.init(project="infopath", config=opt)
    # opt.wandb_run_id = wandb.run.id
    train(opt, model, netD, optimizerG, optimizerD)
