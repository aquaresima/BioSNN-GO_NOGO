import json
import numpy as np
import torch
from tqdm import tqdm
from infopath.utils.functions import trial_match_template, return_trial_type
import pandas as pd

LOG_PATH = "log_dir/main/2026_6_5_6_21_35_full"

def light_generator(time, window="baseline", freq=100):
    time = torch.tensor(time)
    t = torch.arange(time.shape[0]) * 2 / 1000
    light = torch.sin(2 * t * torch.pi * freq) >= 0
    if window == "baseline":
        main = (-1 <= time) & (time <= -0.2)
        ramp = (-0.2 < time) & (time <= -0.1)
    elif window == "whisker":
        main = (-0.1 <= time) & (time <= 0.1)
        ramp = (0.1 < time) & (time <= 0.2)
    elif window == "delay":
        main = (0.2 <= time) & (time <= 0.9)
        ramp = (0.9 < time) & (time <= 1)
    elif window == "response":
        main = (1 <= time) & (time <= 2)
        ramp = (2 < time) & (time <= 2.1)
    else:
        assert False, "window name wrong"
    main = main * 1.0
    start_ramp = (ramp * 1.0).argmax()
    stop_ramp = start_ramp + ramp.sum()
    t = torch.arange(time.shape[0])
    if stop_ramp != start_ramp:
        ramp = ramp * (stop_ramp - t) / (stop_ramp - start_ramp)
        total = main + ramp
    else:
        total = main
    light = light * total
    return light, total


def light_area(light, batch_size, model, area):
    assert area < model.opt.num_areas, "wrong area"
    device = model.opt.device
    light_source = torch.zeros(
        model.opt.n_units, batch_size, light.shape[0], device=device
    )
    light = light.to(model.opt.device)
    light_source[model.rsnn.area_index == area, :, :] = light
    light_source = light_source.permute(2, 1, 0) * 20 * model.rsnn.base_thr[0]
    return light_source


def opto_effect(
    time_vector,
    filt_data,
    filt_model,
    session_info,
    model,
    stims,
    input_spikes,
    state,
    mem_noise,
    filt_data_jaw=None,
    filt_jaw=None,
    lick_detector=None,
    response_time=None,
    seed=0,
    verbose=True,
    power=0.05,
    with_dt=True,
):
    trial_types, trial_types_perc = return_trial_type(
        model,
        filt_data,
        filt_model,
        filt_data_jaw,
        filt_jaw,
        session_info,
        stims,
        lick_detector,
        response_time=response_time,
    )
    miss_p = lambda x: (x == 0).sum() / torch.isin(x, torch.tensor([0, 1])).sum()
    miss_perc = miss_p(trial_types.cpu())

    if verbose:
        print(trial_types_perc)
    fun = model.step_with_dt if with_dt else model.step

    # set the input and noise
    df = pd.DataFrame(columns=("period", "area", "miss_diff", "propabilities"))
    for opto_area in range(model.opt.num_areas):
        df_entry = {
            "period": ["off"],
            "area": [model.opt.areas[opto_area]],
            "miss_diff": [torch.tensor(0).item()],
            "propabilities": [trial_types_perc.cpu().numpy().tolist()],
        }
        df_entry = pd.DataFrame(df_entry)
        df = pd.concat([df, df_entry], ignore_index=True)
        for period in ["whisker", "delay", "response"]:
            # run network with light
            light, envelope = light_generator(time_vector, period)
            envelope = model.filter_fun1(envelope[:, None, None])[:, 0, 0]
            light = light_area(light, model.opt.batch_size, model, opto_area)
            torch.manual_seed(seed)
            if type(power) == list:
                p = power[opto_area]
            else:
                p = power
            spikes_light, _, jaw_light, _ = fun(
                input_spikes, state, mem_noise, light=light * p, dt=50
            )

            filt_model_light = model.filter_fun1(spikes_light)
            filt_jaw_light = model.filter_fun1(jaw_light)

            trial_types_light, trial_types_light_perc = return_trial_type(
                model,
                filt_data,
                filt_model_light,
                filt_data_jaw,
                filt_jaw_light,
                session_info,
                stims,
                lick_detector,
                response_time=response_time,
            )
            if verbose:
                print(opto_area, period, trial_types_light_perc)
            miss_perc_light = miss_p(trial_types_light.cpu())
            df_entry["area"] = [model.opt.areas[opto_area]]
            df_entry["period"] = [period]
            df_entry["propabilities"] = [trial_types_light_perc.tolist()]
            df_entry["miss_diff"] = [(miss_perc_light - miss_perc).cpu()]
            df_entry = pd.DataFrame(df_entry)
            df = pd.concat([df, df_entry], ignore_index=True)
    return df


def set_opto_population(model, area_id, excite=True):
    """
    Selects which neurons receive light.
    excite=True  -> excitatory
    excite=False -> inhibitory
    """
    return (
        (model.rsnn.area_index == area_id)
        & (model.rsnn.excitatory_index == excite)
    )


def run_light_tjm1_response(
    time_vector,
    model,
    stims,
    power,
    filt_data_test,
    filt_jaw_test,
    session_info_test,
    lick_classifier,
    seed=0,
):
    light, envelope = light_generator(time_vector, "response")
    envelope = model.filter_fun1(envelope[:, None, None])[:, 0, 0]
    light = light_area(light, model.opt.batch_size, model, 5)
    mem_noise = model.rsnn.mem_noise.clone()
    trial_noise = model.rsnn.trial_noise.clone()
    torch.manual_seed(seed)
    state = model.steady_state()
    input_spikes = model.input_spikes(stims)
    model.rsnn.mem_noise = mem_noise
    model.rsnn.trial_noise = trial_noise
    with torch.no_grad():
        torch.manual_seed(seed)
        spikes_light, _, jaw_light, _ = model.step_with_dt(
            input_spikes,
            state,
            light=light * power,
            sample_mem_noise=False,
            dt=50,
        )
        filt_model_light = model.filter_fun1(spikes_light) / model.timestep
        filt_jaw_light = model.filter_fun1(jaw_light)

        trial_types_light, trial_types_light_perc = return_trial_type(
            model,
            filt_data_test,
            filt_model_light,
            filt_jaw_test,
            filt_jaw_light,
            session_info_test,
            stims,
            lick_classifier,
        )
        hr = (trial_types_light == 1).sum() / (trial_types_light < 2).sum()
        far = (trial_types_light == 3).sum() / (trial_types_light >= 2).sum()
    return hr, far


def tune_power(
    time_vector,
    model,
    stims,
    filt_data_test,
    filt_jaw_test,
    session_info_test,
    lick_classifier,
    HR_light_tjM1_response,
    seed=0,
):
    with torch.no_grad():
        hr_diff = []
        powers = [0.05, 0.07, 0.1, 0.15]
        for power in powers:
            hr, far = run_light_tjm1_response(
                time_vector,
                model,
                stims,
                power,
                filt_data_test,
                filt_jaw_test,
                session_info_test,
                lick_classifier,
                seed=seed,
            )
            hr_diff.append(hr - HR_light_tjM1_response)
        hr_diff = torch.stack(hr_diff)
        power_id = torch.argmin(hr_diff.abs())
        return powers[power_id]

def matrix_analysis(model):
    W = model.rsnn._w_rec.detach().cpu().squeeze()
    print("W shape:", W.shape)

    binary = (W.abs() > 1e-5).numpy().astype(int)
    pd.DataFrame(binary).to_csv("W_binary.csv", index=False)

    off_diag = model.rsnn._w_rec[:, model.rsnn.off_diag]
    stats = {
        "global_sparsity": ((W.abs()>1e-5).float().mean()).item(),
        "inter_area_sparsity": ((off_diag.abs()>1e-5).float().mean()).item(),
        "inter_area_density": ((off_diag > 1e-5).sum() / off_diag.numel()).item()
    }

    json.dump(stats, open("network_stats.json", "w"), indent=4)

    areas = model.rsnn.area_index.cpu()
    area_names = model.opt.areas
    n_areas = len(area_names)

    mean_matrix = np.zeros((n_areas, n_areas))
    density_matrix = np.zeros((n_areas, n_areas))
    abs_matrix = np.zeros((n_areas, n_areas))

    for src in range(n_areas):
        src_idx = areas == src
        for dst in range(n_areas):
            dst_idx = areas == dst

            block = W[dst_idx][:, src_idx]
            mean_matrix[dst, src] = (block.abs()>1e-5).float().mean().item()
            density_matrix[dst, src] = (block.abs()>1e-5).float().mean().item()
            abs_matrix[dst, src] = block.abs().mean().item()

    pd.DataFrame(mean_matrix, index=area_names, columns=area_names).to_csv("area_mean_weights.csv")
    pd.DataFrame(abs_matrix, index=area_names, columns=area_names).to_csv("area_abs_weights.csv")
    pd.DataFrame(density_matrix, index=area_names, columns=area_names).to_csv("area_density.csv")


if __name__ == "__main__":
    from infopath.utils.functions import load_data
    from infopath.lick_classifier import prepare_classifier
    from infopath.config import load_training_opt
    from infopath.model_loader import load_model_and_optimizer


    opt = load_training_opt(LOG_PATH)
    opt.device = "cuda"

    model = load_model_and_optimizer(opt, reload=True, last_best="best")[0]
    # model = load_model_and_optimizer(opt, reload=True, last_best="best_trial_type_t_trial_ratio")[0]
    # model = load_model_and_optimizer(opt, reload=True, last_best="best_t_trial_ratio")[0]
    print("Model loaded")

    matrix_analysis(model)

    (train_spikes, train_jaw, session_info_train, test_spikes, test_jaw, session_info_test,) = load_data(model)
    print("Data loaded")

    filt_jaw_train = model.filter_fun1(train_jaw)
    filt_jaw_test = model.filter_fun1(test_jaw)

    lick_classifier = prepare_classifier(
        model.filter_fun2(filt_jaw_train),
        model.filter_fun2(filt_jaw_test),
        session_info_train,
        session_info_test,
        opt.device,
    )

    model.rsnn.light_neuron = (1 - model.rsnn.excitatory_index.long()).bool()
    batch_size = 200
    trials = 200

    model.opt.batch_size = batch_size
    time_vector = np.arange(model.T)*0.002 - 0.15
    dfs = []
    
    seeds = np.arange(0, 9)
    for seed in tqdm(seeds):
        print(f"Running seed {seed}")
        torch.manual_seed(seed)
        stims = torch.randint(2, size=(trials,), device=model.opt.device,)
        state = model.steady_state()
        input_spikes = model.input_spikes(stims)
        model.rsnn.sample_mem_noise(model.T, trials)
        mem_noise = model.rsnn.mem_noise.clone()
        model.rsnn.sample_trial_noise(trials)

        with torch.no_grad():
            spikes, voltages, jaw, _ = model.step_with_dt(
                input_spikes, 
                state,
                mem_noise, 
                dt=50
            )

            filt_model = model.filter_fun1(spikes)
            df = opto_effect(
                time_vector, 
                model.filter_fun1(train_spikes),
                filt_model, 
                session_info_train,
                model,
                stims,
                input_spikes,
                state,
                mem_noise,
                filt_data_jaw=model.filter_fun1(train_jaw),
                filt_jaw=model.filter_fun1(jaw),
                lick_detector=lick_classifier,
                seed=seed,
                verbose=True,
                power=0.2,
            )
            dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)
    output_file = "opto_results.csv"
    dfs.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
