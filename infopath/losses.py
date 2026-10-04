import numpy as np
import torch
from infopath.utils.functions import (
    feature_pop_svd,
    mse_2d,
    session_tensor,
    mse_2dv2,
    feature_pop_avg,
)
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import poisson
from torch.autograd import grad


def firing_rate_loss(data_spikes, model_spikes, trial_onset, timestep):
    """Calculate the firing rate loss, the firing rate is calculated from the baseline activity."""
    data_fr = data_spikes[:trial_onset].nanmean(0).nanmean(0) / timestep
    model_fr = model_spikes[:trial_onset].mean(0).mean(0) / timestep
    return ((model_fr - data_fr) ** 2).mean()


def hard_trial_matching_loss(filt_data_spikes, filt_model_spikes):
    """Calculate the trial-matching loss with the hard matching (Hungarian Algorithm)
    Args:
        filt_data_spikes (torch.tensor): $\mathcal{T}_{trial}(z^\mathcal{D})$, dim Trials x Time
        filt_model_spikes (torch.tensor): $\mathcal{T}_{trial}(z)$, dim Trials x Time
    """
    # downsample the biggest tensor, so both data and model have the same #trials
    with torch.no_grad():
        cost = mse_2d(filt_model_spikes.T, filt_data_spikes.T)
        keepx, ytox = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.nn.MSELoss()(filt_model_spikes[keepx], filt_data_spikes[ytox])


def hard_trial_matching_loss_single(feat_data, feat_model, filt_data, filt_model):
    """Calculate the trial-matching loss with the hard matching (Hungarian Algorithm)
    Args:
        filt_data_spikes (torch.tensor): $\mathcal{T}_{trial}(z^\mathcal{D})$, dim Trials x Time
        filt_model_spikes (torch.tensor): $\mathcal{T}_{trial}(z)$, dim Trials x Time
    """
    # downsample the biggest tensor, so both data and model have the same #trials
    min_trials = min(feat_model.shape[0], feat_data.shape[0])
    feat_data = feat_data[:min_trials]
    feat_model = feat_model[:min_trials]
    filt_model = filt_model[:min_trials]
    filt_data = filt_data[:min_trials]
    with torch.no_grad():
        cost = mse_2d(feat_model.T, feat_data.T)
        keepx, ytox = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.nn.MSELoss()(filt_model[:, keepx], filt_data[:, ytox])


def trial_matched_mle_loss(
    data_spikes, model_spikes, session_info, data_jaw, model_jaw
):
    """Calculate the trial matched MLE, see Appendix figure 2"""
    loss = 0
    for session in range(len(session_info[0])):
        (data_spikes_sess, model_spikes_sess, _, _, _) = session_tensor(
            session, session_info, data_spikes, model_spikes, data_jaw, model_jaw
        )
        data_spikes_sess = (data_spikes_sess > 0) * 1.0
        mean = data_spikes_sess.mean((0, 1))
        std = data_spikes_sess.std((0, 1))
        data_spikes_sess = (data_spikes_sess - mean) / std
        model_spikes_sess = (model_spikes_sess - mean) / std
        # downsample the biggest tensor, so both data and model have the same #trials
        min_trials = min(model_spikes_sess.shape[1], data_spikes_sess.shape[1])
        T, K, N = model_spikes_sess.shape
        model_spikes_sess = model_spikes_sess[
            :, torch.randperm(K)[:min_trials]
        ].permute(1, 0, 2)
        T, K, N = data_spikes_sess.shape
        data_spikes_sess = data_spikes_sess[:, torch.randperm(K)[:min_trials]].permute(
            1, 0, 2
        )
        with torch.no_grad():
            cost = mse_2dv2(
                model_spikes_sess.permute(1, 0, 2),
                data_spikes_sess.permute(1, 0, 2),
            )
            keepx, ytox = linear_sum_assignment(cost.detach().cpu().numpy())
        loss += nn.MSELoss()(model_spikes_sess[:, keepx], data_spikes_sess[:, ytox])
    return loss


def roll_dim1(x, d, max_l):
    if d >= 0 and d < max_l:
        x = x[:, d : -(max_l - d)]
    elif max_l == d:
        x = x[:, d:]
    else:
        x = x[:, :-max_l]
    return x


def noise_cross_corr(spikes, l, delay, max_l, norm=None, full_average=True):
    K, T, N = spikes.shape
    T = T - max_l
    if norm is None:
        C = spikes.mean((0, 1))
        norm = torch.sqrt(C[:, None] @ C[None])
    spikes_shift = roll_dim1(spikes - l, -delay, max_l)
    spikes = roll_dim1(spikes, delay, max_l)
    if full_average:
        C_noise = torch.einsum("kti, ktj -> ij", spikes, spikes_shift)
    else:
        C_noise = torch.einsum("kti, ktj -> kij", spikes, spikes_shift)
    return C_noise / (norm + 1e-6)


def cross_corr_guillaume(spikes, delay, filt_fun, norm=None):
    K, N, T = spikes.shape
    spikes_shift = torch.roll(spikes, delay, dims=2)
    spikes_shift_norm = spikes_shift - filt_fun(spikes_shift)
    corr = torch.einsum("kit, kjt -> ij", spikes, spikes_shift_norm) / K / T
    if norm is None:
        C = spikes.mean((0, 2))
        norm = torch.sqrt(C[:, None] @ C[None])
    return corr / (norm + 1e-6)


def cross_corr_jitter(spikes, delay, filt_fun):

    normalizer_pre = spikes.sum((0, 2))[:, None].clamp(min=1e-6)
    normalizer_post = normalizer_pre.T
    normalizer = torch.sqrt(normalizer_pre * normalizer_post)

    K, N, T = spikes.shape
    spikes_shift = torch.roll(spikes, delay, dims=2)
    corr = torch.einsum("ijk, ilk -> jl", spikes, spikes_shift)
    CCG = corr

    k, n, t = torch.where(spikes > 0)
    t += torch.randint(-12, 12, (len(t),), device=spikes.device)
    t = t.clamp(0, T - 1)
    spikes = torch.zeros_like(spikes)
    spikes[k, n, t] = 1
    spikes_shift = torch.roll(spikes, delay, dims=2)
    corr = torch.einsum("ijk, ilk -> jl", spikes, spikes_shift)
    CCG_jitter = corr

    return (CCG - CCG_jitter) / normalizer


@torch.no_grad()
def reorder_model_trials(model, spikes_model, spikes_data, area_index, exc_index):
    sp_m = model.filter_fun2(model.filter_fun1(spikes_model))
    sp_d = model.filter_fun2(model.filter_fun1(spikes_data))
    feat_data, feat_model = feature_pop_avg(
        sp_d, sp_m, None, None, area_index, exc_index, 0
    )
    cost = mse_2d(feat_model.T, feat_data.T)
    keepx, ytox = linear_sum_assignment(cost.detach().cpu().numpy())
    return keepx, ytox


def cross_corr_loss(
    model,
    spikes_data_all,
    spikes_model_all,
    session_info,
    window=0.008,
    full_average=False,
):
    """Calculate the noise cross-correlation loss. Not used in this paper. Similar to pub-bellec-wang-2021 (1)"""
    max_dur = int(window / model.timestep)
    T, K, N = spikes_data_all.shape
    total_cc = torch.zeros(max_dur - 1, model.opt.n_units, model.opt.n_units)
    total_cc = total_cc.to(model.opt.device)
    for session in range(len(session_info[0])):
        spikes_data, spikes_model, _, _, idx = session_tensor(
            session, session_info, spikes_data_all, spikes_model_all, None, None
        )
        min_tr = min(spikes_data.shape[1], spikes_model.shape[1])
        tr_shuffle = torch.randperm(spikes_data.shape[1])
        spikes_data = spikes_data[:, tr_shuffle][:, :min_tr]
        trial_types = session_info[0][session][tr_shuffle][:min_tr]
        tr_shuffle = torch.randperm(spikes_model.shape[1])
        spikes_model = spikes_model[:, tr_shuffle][:, :min_tr]
        _, ytox = reorder_model_trials(
            model,
            spikes_model,
            spikes_data,
            model.rsnn.area_index[idx],
            model.rsnn.excitatory_index[idx],
        )
        c_model = calculate_cross_corr(
            model, spikes_model, 1, max_dur, trial_types[ytox], torch.ones(N).bool()
        )[0]
        total_cc = c_model
    mask = model.cc_data.sum(0) == 0
    cross_corr_loss = (
        (model.cc_data[1:3, ~mask].mean(0) - total_cc[1:3, ~mask].mean(0)) ** 2
    ).mean()
    return cross_corr_loss, c_model.var(1).mean() / c_model.shape[1]


def calculate_cross_corr_per_trial(spikes, start, stop, trial_types):
    T, K, N = spikes.shape
    ks = [0]
    for tt in np.unique(trial_types):
        tt_d = trial_types == tt
        if (tt_d * 1.0).mean() > 0.05:
            ks.append(tt_d.sum().item() + ks[-1])
    c_data = torch.zeros(stop - start, ks[-1], N, N).cuda()
    with torch.no_grad():
        norm = torch.sqrt(
            spikes.var((0, 1))[:, None] @ spikes.var((0, 1))[None] * T * K
        )
        norm[norm == 0] = 1
        norm = 1
    j = 0
    for tt in np.unique(trial_types):
        tt_d = trial_types == tt
        if (tt_d * 1.0).mean() < 0.05:
            continue
        l_d = spikes[:, tt_d].mean(1)
        for i, d in enumerate(range(start, stop)):
            signal = spikes[:, tt_d].permute(1, 0, 2)
            c_data[i, ks[j] : ks[j + 1]] += noise_cross_corr(
                signal,
                l_d,
                d,
                max(abs(start), abs(stop)),
                norm=norm,
                full_average=False,
            )
        j += 1
    c_data[:, :, torch.eye(N).bool()] = 0
    return c_data


def calculate_cross_corr(model, spikes, start, stop, trial_types, idx, cZeros=None):
    device = spikes.device
    T, K, N = spikes.shape
    if cZeros is None:
        cZeros = torch.ones(N, N).bool().to(device)
    c_data = torch.zeros(stop - start, N, N)
    c_data = c_data.to(device)
    idx = torch.tensor(idx, device=device)
    sess_neurons = (idx[:, None].float() @ idx[None].float()).bool()
    filt_size = int(0.023 / model.timestep)
    filt_size = filt_size + 1 if filt_size % 2 == 0 else filt_size
    padding = filt_size // 2
    filt_fun = torch.nn.Conv1d(
        N,
        N,
        kernel_size=filt_size,
        padding=padding,
        bias=False,
        groups=N,
        padding_mode="reflect",
    )
    kernel = torch.ones(filt_size)
    kernel[padding + 1] = 0.6
    filt_fun.weight.data[:] = kernel[None, None] / kernel.sum()
    filt_fun.to(spikes.device)
    with torch.no_grad():
        norm = torch.sqrt(
            spikes.mean((0, 1))[:, None] @ spikes.mean((0, 1))[None] * T * K
        )
        norm[~cZeros[idx, :][:, idx]] = 1
        norm = 1
    if model.opt.cc_version == 0:
        for tt in np.unique(trial_types):
            tt_d = trial_types == tt
            if (tt_d * 1.0).mean() < 0.05:
                continue
            l_d = spikes[:, tt_d].mean(1)
            tt_ratio = tt_d.sum() / spikes.shape[1]
            for i, d in enumerate(range(start, stop)):
                signal = spikes[:, tt_d].permute(1, 0, 2)
                c_data[i, sess_neurons] += (
                    noise_cross_corr(
                        signal, l_d, d, max(abs(start), abs(stop)), norm=norm
                    ).flatten()
                    * tt_ratio
                )
                c_data[i].fill_diagonal_(0)
                c_data[i, sess_neurons] *= cZeros[sess_neurons]
    else:
        for i, d in enumerate(range(start, stop)):
            signal = spikes.permute(1, 0, 2)
            with torch.no_grad():
                l = filt_fun(signal.permute(0, 2, 1)).permute(0, 2, 1)
            c_data[i, sess_neurons] += noise_cross_corr(
                signal, l, d, max(abs(start), abs(stop)), norm=norm
            ).flatten()
            c_data[i].fill_diagonal_(0)
            c_data[i, sess_neurons] *= cZeros[sess_neurons]
    c_data = c_data / (K) ** 0.5
    return c_data, sess_neurons


interval = (
    lambda fr, T, K: (poisson.ppf(1 - 1e-4, fr * T * K) - fr * T * K)
    / (T * K)
    # / fr ** 0.5
)


def find_important_elements(model, spikes_data_all, session_info, window=0.008):
    s = int(window / model.timestep)
    total_cc = torch.zeros(s - 1, model.opt.n_units, model.opt.n_units)
    total_cc = total_cc.to(model.opt.device)
    for session in range(len(session_info[0])):
        spikes_data, _, _, _, idx = session_tensor(
            session, session_info, spikes_data_all, None, None, None
        )
        if spikes_data.shape[1] < 100:
            continue
        c_data, sess_neurons = calculate_cross_corr(
            model, spikes_data, 1, s, session_info[0][session], idx
        )
        T, K, N = spikes_data.shape
        # cmax = c_data.abs().max(0)[0] / (K) ** 0.5
        # strong_cc = cmax > 1
        # total_cc[:, sess_neurons & strong_cc] = c_data[:, sess_neurons & strong_cc]
        total_cc += c_data
    model.cc_data = total_cc


def calculate_directionality(spike, trial_type, model, size=52, window=10):
    N = spike.shape[2]
    c = torch.zeros(size, N, N).to(spike.device)
    fr = ((spike.mean((0, 1)) / model.timestep) > 1).float()
    cZeros = fr[:, None] @ fr[None]
    for tt in trial_type.unique():
        if (trial_type == tt).sum() < 30:
            continue
        l = spike[:, trial_type == tt].mean(1)
        for i, j in enumerate(range(-size // 2, size // 2)):
            c[i] += noise_cross_corr(
                spike[:, trial_type == tt].clone().permute(1, 0, 2), l, j
            )
            c[i].fill_diagonal_(0)
            c[i] *= cZeros * (trial_type == tt).sum() / trial_type.shape[0]

    cmax_forw = (c[size // 2 + 1 : size // 2 + window]).abs().max(0)[0]
    cmax_back = (c[size // 2 - window + 1 : size // 2]).abs().max(0)[0]
    std = torch.cat([c[size // 2 + window :], c[1 : size // 2 - window + 1]], 0).std(0)
    strong_correlations_forw = cmax_forw > std * 5
    strong_correlations_back = cmax_back > std * 5
    forward = mean_pop_w([strong_correlations_forw * 1.0])
    backward = mean_pop_w([strong_correlations_back * 1.0])
    directionality = (forward - backward) / (forward.sum() + backward.sum() + 1e-6)
    print((forward.sum() + backward.sum()) / 1500 / 1500)
    return directionality


def mean_pop_w(w_rec, num_areas=6):
    area_index = torch.zeros(1500)
    for i in range(num_areas):
        area_index[i * 200 : (i + 1) * 200] = i
        area_index[1200 + 50 * i : 1200 + (i + 1) * 50] = i
    w = torch.zeros(num_areas, num_areas)
    for i in range(num_areas):
        for j in range(num_areas):
            index_in = area_index == i
            index_out = area_index == j
            w[j, i] = w_rec[0][index_out][:, index_in].mean()
    return w


def mean_pop_w_with_ei(w_rec, num_areas=6):
    area_index = torch.zeros(1500)
    for i in range(num_areas):
        area_index[i * 200 : (i + 1) * 200] = i
        area_index[1200 + 50 * i : 1200 + (i + 1) * 50] = i
    exc_index = torch.ones(1500)
    exc_index[1200:] = 0
    w = torch.zeros(num_areas * 2, num_areas * 2)
    for i in range(num_areas):
        for j in range(num_areas):
            index_in_e = (area_index == i) & (exc_index == 1)
            index_out_e = (area_index == j) & (exc_index == 1)
            index_in_i = (area_index == i) & (exc_index == 0)
            index_out_i = (area_index == j) & (exc_index == 0)
            w[j, i] = w_rec[0][index_out_e][:, index_in_e].mean()
            w[j, i + num_areas] = w_rec[0][index_out_e][:, index_in_i].mean()
            w[j + num_areas, i] = w_rec[0][index_out_i][:, index_in_e].mean()
            w[j + num_areas, i + num_areas] = w_rec[0][index_out_i][
                :, index_in_i
            ].mean()
    return w


def perturbation_loss(model, stims, optimizer, spikes_data_all, power=0.1):
    for i in range(model.opt.num_areas):
        optimizer.zero_grad()
        light = (
            torch.ones(model.T, stims.shape[0], model.opt.n_units).to(model.opt.device)
            * power
        )
        inh = model.rsnn.excitatory_index == 0
        model.rsnn.light_neuron = (model.rsnn.area_index == i) & inh
        model_spikes, _, _, _ = model(stims, light=light)
        loss = 0
        for j in range(model.opt.num_areas):
            if i == j:
                continue
            area_index = model.rsnn.area_index == j
            loss += nn.MSELoss()(
                model_spikes[:, :, area_index].mean((1, 2)),
                spikes_data_all[:, :, area_index].nanmean((1, 2)),
            )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.rsnn.reform_recurent(optimizer.param_groups[0]["lr"], l1_decay=0)


def z_score_norm(data_signal, model_signal, dim=1):
    # initial dim (x,y,z) sig = (sig-min)/(max-min), final dim (x,(!dim))
    data_meandim = data_signal.nanmean(dim)
    model_meandim = model_signal.nanmean(dim)
    mean = data_meandim.mean(0)
    std = data_meandim.std(0)
    std.clip_(1e-3)
    model_meandim = (model_meandim - mean) / std
    data_meandim = (data_meandim - mean) / std
    return model_meandim, data_meandim, (model_signal - mean) / std


def gradient_penalty(feat_data, feat_model, netD, session, t_trial_gan):
    if t_trial_gan:
        epsilon = torch.rand(feat_data.shape[0], 1, device=feat_data.device)
    else:
        feat_data = feat_data.permute(1, 2, 0)
        feat_model = feat_model.permute(1, 2, 0)
        epsilon = torch.rand(feat_data.shape[0], 1, 1, device=feat_data.device)
    x_hat = feat_data * epsilon + feat_model * (1 - epsilon)
    x_hat.requires_grad_(True)
    out_x_hat = netD.discriminators[session](x_hat)
    gradients = torch.autograd.grad(
        outputs=out_x_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(out_x_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    dims = [1] if t_trial_gan else [1, 2]
    return ((gradients.norm(2, dim=dims) - 1) ** 2).mean()


def discriminator_loss(
    netD,
    model_spikes,
    data_spikes,
    data_jaw,
    model_jaw,
    session_info,
    area_index,
    exc_index,
    discriminator=True,
    t_trial_gan=True,
    z_score=True,
):
    gp = 0  # gradient penalty
    output_data, output_model, labels_data, labels_model = [], [], [], []
    for session in range(len(session_info[0])):
        (
            filt_data_s,
            filt_model_s,
            f_data_jaw,
            f_model_jaw,
            idx,
        ) = session_tensor(
            session,
            session_info,
            data_spikes,
            model_spikes,
            data_jaw,
            model_jaw,
        )
        min_trials = min(filt_model_s.shape[1], filt_data_s.shape[1])
        perm_data = torch.randperm(filt_data_s.shape[1])[:min_trials]
        filt_data_s = filt_data_s[:, perm_data]
        perm_model = torch.randperm(filt_model_s.shape[1])[:min_trials]
        filt_model_s = filt_model_s[:, perm_model]
        if f_data_jaw is not None:
            f_data_jaw = f_data_jaw[:, perm_data]
            f_model_jaw = f_model_jaw[:, perm_model]
        if discriminator:
            netD.train()
        else:
            netD.eval()
        if t_trial_gan:
            feat_data, feat_model = feature_pop_avg(
                filt_data_s,
                filt_model_s,
                f_data_jaw,
                f_model_jaw,
                area_index[idx],
                exc_index[idx],
                session,
                z_score=z_score,
            )
            out_data = netD.discriminators[session](feat_data)
            out_model = netD.discriminators[session](feat_model)
        else:
            feat_data = filt_data_s
            feat_model = filt_model_s
            if f_data_jaw is not None:
                feat_data = torch.cat([feat_data, f_data_jaw])
                feat_model = torch.cat([feat_model, f_model_jaw])
            inp = torch.cat([feat_data, feat_model], 1).permute(1, 2, 0)
            out = netD.discriminators[session](inp)
            out_data = out[:min_trials]
            out_model = out[min_trials:]
        if len(feat_data) == 0:
            continue
        l_data = torch.ones_like(out_data)
        if discriminator:
            l_model = torch.zeros_like(out_model)
            gp += gradient_penalty(feat_data, feat_model, netD, session, t_trial_gan)
        else:
            l_model = torch.ones_like(out_model)
        output_data.append(out_data)
        output_model.append(out_model)
        labels_data.append(l_data)
        labels_model.append(l_model)
    output_data = torch.cat(output_data)
    output_model = torch.cat(output_model)
    labels_model = torch.cat(labels_model)
    labels_data = torch.cat(labels_data)

    if discriminator:
        loss = output_model.mean() - output_data.mean() + gp * 10
        # loss = (bce(output_data, labels_data) + bce(output_model, labels_model)) / 2
    else:
        loss = -output_model.mean()
        # loss = bce(output_model, labels_model)
    labels = torch.cat((labels_model, labels_data))
    out = torch.cat((output_model, output_data))
    accuracy = (((out > 0.5) == (labels > 0)) * 1.0).mean()
    return loss, accuracy


def trial_matching_loss(
    model,
    filt_data_all,
    filt_model_all,
    session_info,
    data_jaw,
    model_jaw,
    loss_fun,
    area_index,
    exc_index,
    z_score=True,
    trial_loss_area_specific=True,
    trial_loss_exc_specific=False,
    feat_svd=False,
    dim=10,
    stims=None,
):
    """Here we actually calculate the trial-matching loss function

    If `stims` (stimulus 0/1 of every model trial) is given, the matching is stratified: model trials
    are only compared with data trials of the same stimulus (session_info[1]), one loss term per
    (session, stimulus) group. Without it, matching ignores the stimulus (original behaviour).

    Args:
        filt_data_all (torch.tensor): filtered data spikes
        filt_model_all (torch.tensor): filtered model spikes
        session_info (list): information about different sessions
        data_jaw (torch.tensor): filtered data jaw trace
        model_jaw (torch.tensor): filtered model jaw trace
        loss_fun (function): function either hard_trial_matching_loss or sinkhorn loss
        area_index (torch.tensor): tensor with areas
        z_score (bool, optional): whether to z_score or not. Defaults to True.
        trial_loss_area_specific (bool, optional): Whether the T_trial is area specific. Defaults to True.

    Returns:
        _type_: _description_
    """
    loss, sessions = 0, 0
    # here we choose if we use soft or hard trial matching
    for session in range(len(session_info[0])):
        filt_data, filt_model, f_data_jaw, f_model_jaw, idx = session_tensor(
            session,
            session_info,
            filt_data_all,
            filt_model_all,
            data_jaw,
            model_jaw,
        )
        # if the session has less than 10 neurons or no trials(this happens only for
        #   particular trial type & stimulus conditions) don't take into account
        if filt_data.shape[2] < 10 or filt_model.shape[1] == 0:
            continue
        if feat_svd:
            feat_data, feat_model = feature_pop_svd(
                model, filt_data, filt_model, 0, 0, 0, 0, 0, dim=dim
            )
        else:
            feat_data, feat_model = feature_pop_avg(
                filt_data,
                filt_model,
                f_data_jaw,
                f_model_jaw,
                area_index[idx],
                exc_index[idx],
                session,
                z_score=z_score,
                trial_loss_area_specific=trial_loss_area_specific,
                trial_loss_exc_specific=trial_loss_exc_specific,
            )
        if stims is None:
            groups = [(feat_data, feat_model)]
        else:
            data_stim = torch.as_tensor(session_info[1][session]).to(feat_data.device)
            model_stim = torch.as_tensor(stims).to(feat_model.device)
            assert data_stim.shape[0] == feat_data.shape[0], "session_info stims do not match the data trials"
            assert model_stim.shape[0] == feat_model.shape[0], "stims do not match the model trials"
            groups = [
                (feat_data[data_stim == g], feat_model[model_stim == g]) for g in (0, 1)
            ]
        for fd, fm in groups:
            min_trials = min(fm.shape[0], fd.shape[0])
            if min_trials == 0:
                continue
            fd = fd[torch.randperm(fd.shape[0])[:min_trials]]
            fm = fm[torch.randperm(fm.shape[0])[:min_trials]]
            # apply the proper scaling depending the method
            scaling = 1
            if loss_fun.__dict__ != {}:
                if loss_fun.loss != "energy":
                    scaling = fd.shape[1]
                if loss_fun.p == 1:
                    scaling = scaling**0.5
                scaling = fd.shape[1] ** 0.5
            loss += loss_fun(fd, fm) / scaling
            sessions += 1
    loss /= sessions
    return loss


def data_stim_tensor(session_info, n_trials, n_neurons, device):
    """Stimulus (0/1) of every data trial, per neuron: (trials, neurons), -1 where the neuron's session has no such trial.

    Data trial k of session s belongs to session_info[1][s][k]; neurons of the session are session_info[-1][s].
    """
    out = torch.full((n_trials, n_neurons), -1, dtype=torch.long, device=device)
    for sess, idx in enumerate(session_info[-1]):
        st = torch.as_tensor(session_info[1][sess]).long().to(device)
        mask = torch.as_tensor(idx).bool().to(device)
        sub = out[: st.shape[0]]
        sub[:, mask] = st[:, None].expand(-1, int(mask.sum()))
    return out


def stratified_psth_loss(f_data, f_model, data_stim, model_stim):
    """Neuron-wise PSTH loss computed separately for the GO and NO-GO stimulus.

    f_data  (T, data trials, neurons), NaN where the neuron's session has no trial
    f_model (T, model trials, neurons)
    data_stim (data trials, neurons) from data_stim_tensor; model_stim (model trials,)
    Normalisation (mean/std per neuron) comes from the all-trial data PSTH, so the GO/NO-GO difference is kept.
    """
    all_mean = f_data.nanmean(1)
    mean = all_mean.mean(0)
    std = all_mean.std(0).clip(1e-3)
    loss = 0
    for g in (0, 1):
        in_g = (data_stim == g)[None]  # (1, trials, neurons)
        d = torch.where(in_g, f_data, torch.full_like(f_data, float("nan"))).nanmean(1)
        m = f_model[:, model_stim == g].nanmean(1)
        loss = loss + torch.nanmean(((m - mean) / std - (d - mean) / std) ** 2)
    return loss / 2


def kl_loss(mu, log_var):
    """Calculate the KL loss for the VAE"""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


def reconstruction_loss_mse(y, y_hat):
    """Calculate the reconstruction loss for the VAE"""
    return torch.nn.MSELoss(reduction="sum")(y, y_hat)


def reconstruction_loss_bce(y, y_hat):
    return torch.nn.BCELoss(reduction="sum")(y_hat, y)


def reconstruction_loss_poisson(y, y_hat):
    return torch.nn.PoissonNLLLoss(reduction="sum")(y, y_hat)


def reconstruction_loss_poisson_v1(y, y_hat):
    nll_all = log_poisson_loss(y, y_hat, compute_full_loss=True)
    return nll_all.mean()


def log_poisson_loss(targets, log_input, compute_full_loss):
    if targets.size() != log_input.size():
        raise ValueError(
            "log_input and targets must have the same shape (%s vs %s)"
            % (log_input.size(), targets.size())
        )

    result = torch.exp(log_input) - log_input * targets
    if compute_full_loss:
        point_five = 0.5
        two_pi = 2 * 3.1415926
        stirling_approx = (
            (targets * torch.log(targets))
            - targets
            + (point_five * torch.log(two_pi * targets))
        )
        zeros = torch.zeros_like(targets, dtype=targets.dtype)
        ones = torch.ones_like(targets, dtype=targets.dtype)
        cond = (targets >= zeros) & (targets <= ones)
        result += torch.where(cond, zeros, stirling_approx)
    return result


def calculate_big_cc_matrix(
    model, trials=5 * 3600, trials_per_round=6000, window=6, thr=7
):
    rounds = trials // trials_per_round
    torch.cuda.empty_cache()
    # we know there are 2 trial types
    mean_trial_activity = torch.zeros(model.T, 2, model.rsnn.n_units).cuda()
    for seed in range(1, rounds + 1):
        with torch.no_grad():
            stims = (
                torch.ones(trials_per_round) * 4
            )  # binary vector of conditions (absence or presence of whisker stimulation)
            torch.manual_seed(seed)
            spikes_m1_orig, _, _, _ = model(stims)
            filt = model.filter_fun2(model.filter_fun1(spikes_m1_orig))
            area0_active = (
                filt[:, :, model.rsnn.area_index == 0].mean((2)).max(0)[0]
                / model.timestep
                > thr
            )
            area1_active = (
                filt[:, :, model.rsnn.area_index == 1].mean((2)).max(0)[0]
                / model.timestep
                > thr
            )
            trial_type_orig_m1 = area0_active * 2 + area1_active
            mean_trial_activity[:, 0] += (
                spikes_m1_orig[:, trial_type_orig_m1 == 0].mean(1) / rounds
            )
            mean_trial_activity[:, 1] += (
                spikes_m1_orig[:, trial_type_orig_m1 == 3].mean(1) / rounds
            )
    # ccg: num_trial_types x num_timepoints x num_units x num_units
    ccg = torch.zeros(7, model.opt.n_units, model.opt.n_units).cuda()
    for seed in range(1, rounds + 1):
        with torch.no_grad():
            torch.cuda.empty_cache()
            stims = (
                torch.ones(trials) * 4
            )  # binary vector of conditions (absence or presence of whisker stimulation)
            torch.manual_seed(seed)
            spikes_m1_orig, _, _, _ = model(stims)
            torch.cuda.empty_cache()
            filt = model.filter_fun2(model.filter_fun1(spikes_m1_orig))
            area0_active = (
                filt[:, :, model.rsnn.area_index == 0].mean((2)).max(0)[0]
                / model.timestep
                > thr
            )
            area1_active = (
                filt[:, :, model.rsnn.area_index == 1].mean((2)).max(0)[0]
                / model.timestep
                > thr
            )
            trial_type_orig_m1 = area0_active * 2 + area1_active
            for i, d in enumerate(range(1, window + 1)):
                tt_d = trial_type_orig_m1 == 0
                tt_ratio = tt_d.sum() / trials
                ccg[i] += (
                    noise_cross_corr(
                        spikes_m1_orig[:, tt_d].permute(1, 0, 2),
                        mean_trial_activity[:, 0],
                        d,
                        window + 1,
                        norm=1,
                    )
                    * tt_ratio
                )
                tt_d = trial_type_orig_m1 == 3
                tt_ratio = tt_d.sum() / trials
                ccg[i] += (
                    noise_cross_corr(
                        spikes_m1_orig[:, tt_d].permute(1, 0, 2),
                        mean_trial_activity[:, 1],
                        d,
                        window + 1,
                        norm=1,
                    )
                    * tt_ratio
                )
                ccg[i].fill_diagonal_(0)
    return ccg / trials**0.5
