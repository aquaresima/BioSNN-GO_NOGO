import numpy as np
import torch

TYPE_NAMES = ["Miss", "Hit", "CR", "FA"]  # dataloader: 0 Miss, 1 Hit, 2 CR, 3 FA
SILENT_HZ = 0.1


def _rates(spikes, timestep, chunk=200):
    """Per-neuron mean rate in Hz from a (T, trials, neurons) tensor; NaN trials are ignored.

    Accumulated over time chunks so no full-size float/mask copy of the tensor is made
    (the test tensor is several GB in the long run).
    """
    total = torch.zeros(spikes.shape[2], dtype=torch.float64, device=spikes.device)
    count = torch.zeros_like(total)
    for a in range(0, spikes.shape[0], chunk):
        x = spikes[a : a + chunk]
        ok = ~torch.isnan(x)
        total += torch.where(ok, x, torch.zeros((), dtype=x.dtype, device=x.device)).sum(dim=(0, 1), dtype=torch.float64)
        count += ok.sum(dim=(0, 1), dtype=torch.float64)
    return (total / count / timestep).float()


def _window(opt, t0, t1, T):
    start = opt.start
    dt = opt.dt / 1000.0
    a = int(np.clip(round((t0 - start) / dt), 0, T))
    b = int(np.clip(round((t1 - start) / dt), 0, T))
    return a, b


def eval_statistics(model, model_spikes, data_spikes, stims, trial_type, model_perc, data_perc):
    """Statistics logged at every evaluation.

    Returns (metrics, confusion, type_table):
      metrics     flat dict of floats (firing rates model vs data, silent fraction, trial-type fractions)
      confusion   2x2 int array, rows = stimulus (NO-GO, GO), columns = class of the assigned trial type
      type_table  2x4 int array, rows = stimulus (NO-GO, GO), columns = assigned type (Miss, Hit, CR, FA)
    """
    ts = model.timestep
    with torch.no_grad():
        fr_m = _rates(model_spikes, ts).cpu()
        fr_d = _rates(data_spikes, ts).cpu()
    valid = ~torch.isnan(fr_d)
    exc = model.rsnn.excitatory_index.cpu().bool()
    area = model.rsnn.area_index.cpu()
    sessions = np.asarray(model.sessions).astype(str)

    metrics = {}

    def add(prefix, mask):
        mask = mask & valid
        if mask.any():
            metrics[f"rate_model/{prefix}"] = fr_m[mask].mean().item()
            metrics[f"rate_data/{prefix}"] = fr_d[mask].mean().item()

    add("all", torch.ones_like(valid))
    add("exc", exc)
    add("inh", ~exc)
    for a, name in enumerate(model.opt.areas):
        add(f"area_{name}", area == a)
    for s in np.unique(sessions):
        if s == "" or s == "0.0":
            continue
        parts = s.split("_")  # "<animal>_<date>_<task>_<area>_<depth>" -> "<animal>_<date>_<area>"
        short = f"{parts[0]}_{parts[1]}_{parts[3]}" if len(parts) >= 4 else s
        add(f"session/{short}", torch.as_tensor(sessions == s))

    # neuron-wise agreement and silent fractions
    both = valid & ~torch.isnan(fr_m)
    if both.sum() > 2:
        x = torch.log10(fr_m[both] + SILENT_HZ)
        y = torch.log10(fr_d[both] + SILENT_HZ)
        metrics["rate_corr_log_model_vs_data"] = float(np.corrcoef(x.numpy(), y.numpy())[0, 1])
        metrics["frac_silent_model"] = (fr_m[both] < SILENT_HZ).float().mean().item()
        metrics["frac_silent_data"] = (fr_d[both] < SILENT_HZ).float().mean().item()

    # task epochs (relative to sound onset): sound 0-2 s, delay 2-4 s, response 4 s on
    T = model_spikes.shape[0]
    for epoch, (t0, t1) in {"sound": (0.0, 2.0), "delay": (2.0, 4.0), "response": (4.0, model.opt.stop)}.items():
        a, b = _window(model.opt, t0, t1, T)
        if b > a:
            m = _rates(model_spikes[a:b], ts).cpu()
            d = _rates(data_spikes[a:b], ts).cpu()
            ok = ~torch.isnan(d)
            metrics[f"rate_model/epoch_{epoch}"] = m[ok].mean().item()
            metrics[f"rate_data/epoch_{epoch}"] = d[ok].mean().item()

    # trial classification
    stims = torch.as_tensor(stims).cpu().long()
    tt = torch.as_tensor(trial_type).cpu().long()
    type_table = np.zeros((2, 4), dtype=int)
    for s in (0, 1):
        for k in range(4):
            type_table[s, k] = int(((stims == s) & (tt == k)).sum())
    confusion = np.zeros((2, 2), dtype=int)
    confusion[:, 0] = type_table[:, 2:].sum(1)  # assigned a NO-GO type (CR, FA)
    confusion[:, 1] = type_table[:, :2].sum(1)  # assigned a GO type (Miss, Hit)
    tot = confusion.sum()
    if tot > 0:
        metrics["stimulus_class_accuracy"] = float(np.trace(confusion) / tot)
    for k, name in enumerate(TYPE_NAMES):
        metrics[f"type_frac_model/{name}"] = float(model_perc[k])
        metrics[f"type_frac_data/{name}"] = float(data_perc[k])
    return metrics, confusion, type_table
