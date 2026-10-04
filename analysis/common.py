"""Shared loading and simulation for the checkpoint analyses."""
import os
import shutil

import numpy as np
import torch

from infopath.config import load_training_opt
from infopath.model_loader import load_model_and_optimizer
from infopath.utils.functions import load_data


def load_checkpoint_and_data(run, out, ckpt="last", n_trials=200):
    """Copy the run's files (a running job may overwrite them), load model, data, and return everything."""
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "run_copy")
    os.makedirs(work, exist_ok=True)
    for f in ["opt.json", f"{ckpt}_model.ckpt", "sessions.npy", "neuron_index.npy", "firing_rate.npy", "areas.npy"]:
        shutil.copy(os.path.join(run, f), work)
    opt = load_training_opt(work)
    opt.log_path = work
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.batch_size = n_trials
    model, _, _, _ = load_model_and_optimizer(opt, reload=True, last_best=ckpt, reload_optim=False)
    model.eval()
    train_spikes, _, info_train, test_spikes, _, info_test = load_data(model)
    return opt, model, train_spikes, info_train, test_spikes, info_test


def simulate(model, opt, n_trials):
    """n_trials simulated trials, half GO (1) and half NO-GO (0). Returns (spikes (T, trials, N), stims (trials,))."""
    stims = torch.tensor([0, 1] * (n_trials // 2), device=opt.device).long()
    with torch.no_grad():
        state = model.steady_state()
        input_spikes = model.input_spikes(stims)
        model.rsnn.sample_mem_noise(input_spikes.shape[0], input_spikes.shape[1])
        mem_noise = model.rsnn.mem_noise.clone()
        spikes, _, _, _ = model.step(input_spikes, state, mem_noise=mem_noise)
    del state, input_spikes
    torch.cuda.empty_cache()
    return spikes, stims


def load_dataset_with_trials(model):
    """Like infopath.utils.functions.load_data, but also returns the TrialDataset (to recover which original
    trials went to train and test: sorted train_ind / test_ind of each session)."""
    from datasets.dataloader import TrialDataset

    opt = model.opt
    dataset = TrialDataset(
        opt.datapath, model.sessions, model.areas, model.neuron_index,
        start=opt.start, stop=opt.stop, stim=opt.stim, trial_type=opt.trial_types,
        reaction_time_limits=opt.reaction_time_limits, timestep=opt.dt * 0.001,
        with_behaviour=opt.with_behaviour, trial_onset=opt.trial_onset, train_perc=0.5,
    )
    dataset.to_torch()
    tr, _, info_tr = dataset.get_train_trial_type(train=1, device=opt.device, jaw_tongue=opt.jaw_tongue)
    te, _, info_te = dataset.get_train_trial_type(train=0, device=opt.device, jaw_tongue=opt.jaw_tongue)
    sessions = list(dataset.data_dict["sessions"])
    train_idx = [np.sort(dataset.data_dict["train_ind"][s]) for s in range(len(sessions))]
    test_idx = [np.sort(dataset.data_dict["test_ind"][s]) for s in range(len(sessions))]
    return tr, info_tr, te, info_te, sessions, train_idx, test_idx
