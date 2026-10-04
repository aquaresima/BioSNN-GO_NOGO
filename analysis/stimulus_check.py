"""Does the model's activity depend on the stimulus the way the data does?

Loads a saved checkpoint, simulates GO and NO-GO trials, and compares with the recordings:
  A. per area and task epoch, population rate for GO / NO-GO trials, model vs data;
  B. per neuron, the GO minus NO-GO rate difference, model vs data (correlation, magnitude);
  C. the template classifier used by the Comet confusion matrix: distances of the model trials
     to the four trial-type templates, and how far the templates are from each other;
  D. PSTH figure (data vs model, GO vs NO-GO, per area).

Usage: python analysis/stimulus_check.py --run log_dir/master/<run> --out analysis/out/<name> [--ckpt last]
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infopath.losses import data_stim_tensor
from analysis.common import load_checkpoint_and_data, simulate
from infopath.utils.eval_stats import TYPE_NAMES, _window
from infopath.utils.functions import make_template

EPOCHS = {"sound": (0.0, 2.0), "delay": (2.0, 4.0), "response": (4.0, None)}


def epoch_means(x, stim_mask, window, timestep):
    """Mean rate (Hz) per neuron over a time window, over trials selected by stim_mask.

    x (T, trials, neurons) with NaN for missing trials; stim_mask (trials, neurons) or (trials,) bool.
    """
    a, b = window
    seg = x[a:b]
    if stim_mask.dim() == 1:
        m = stim_mask[None, :, None]
    else:
        m = stim_mask[None]
    seg = torch.where(m, seg, torch.full_like(seg, float("nan")))
    return (torch.nanmean(seg.mean(0), dim=0) / timestep).cpu()  # mean over time, then trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--n_trials", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    opt, model, train_spikes, info_train, test_spikes, info_test = load_checkpoint_and_data(
        args.run, args.out, args.ckpt, args.n_trials
    )
    ts = model.timestep
    model_spikes, stims = simulate(model, opt, args.n_trials)
    T = model_spikes.shape[0]
    N = model_spikes.shape[2]
    area = model.rsnn.area_index.cpu()
    exc = model.rsnn.excitatory_index.cpu().bool()
    summary = {"run": args.run, "ckpt": args.ckpt, "n_model_trials": args.n_trials}

    # ---- A, B: epoch rates and per-neuron stimulus effect ----------------------------
    dstim = data_stim_tensor(info_train, train_spikes.shape[1], N, train_spikes.device)
    m_stim = stims
    rows = []
    for ename, (t0, t1) in EPOCHS.items():
        w = _window(opt, t0, t1 if t1 is not None else opt.stop, T)
        r = {}
        for g, gname in ((1, "GO"), (0, "NOGO")):
            r[("data", gname)] = epoch_means(train_spikes, dstim == g, w, ts)
            r[("model", gname)] = epoch_means(model_spikes, (m_stim == g), w, ts)
        eff_d = r[("data", "GO")] - r[("data", "NOGO")]
        eff_m = r[("model", "GO")] - r[("model", "NOGO")]
        for a_i, aname in enumerate(opt.areas):
            sel = (area == a_i) & ~torch.isnan(eff_d) & ~torch.isnan(eff_m)
            if sel.sum() < 3:
                continue
            row = {"epoch": ename, "area": aname, "n_neurons": int(sel.sum())}
            for src in ("data", "model"):
                for gname in ("GO", "NOGO"):
                    row[f"rate_{src}_{gname}"] = float(r[(src, gname)][sel].mean())
            row["effect_data_mean_abs"] = float(eff_d[sel].abs().mean())
            row["effect_model_mean_abs"] = float(eff_m[sel].abs().mean())
            row["effect_corr_model_vs_data"] = float(np.corrcoef(eff_d[sel].numpy(), eff_m[sel].numpy())[0, 1])
            rows.append(row)
    summary["epoch_rows"] = rows

    # ---- C: the template classifier -----------------------------------------------
    num_areas = len(opt.areas)
    f_data = model.filter_fun2(model.filter_fun1(test_spikes))
    f_model = model.filter_fun2(model.filter_fun1(model_spikes))
    data_template, mean_area, std_area = make_template(model, f_data, info_test, None, num_areas=num_areas)
    Tt = f_model.shape[0]
    model_template = torch.zeros(num_areas * Tt, f_model.shape[1], device=f_model.device)
    for a_i in range(num_areas):
        tmp = f_model[:, :, model.rsnn.area_index == a_i].sum(2) / 250
        model_template[a_i * Tt : (a_i + 1) * Tt] = ((tmp.T - mean_area[a_i]).T / std_area[a_i])
    dist = torch.stack(
        [((model_template[:, k][None] - data_template.T) ** 2).mean(1) for k in range(f_model.shape[1])]
    ).cpu()  # (trials, 4)
    assigned = dist.argmin(1)
    st = stims.cpu()
    summary["classifier"] = {
        "mean_distance_to_template": {
            g: {TYPE_NAMES[k]: float(dist[st == v, k].mean()) for k in range(4)} for g, v in (("GO", 1), ("NOGO", 0))
        },
        "assigned_counts": {
            g: {TYPE_NAMES[k]: int(((st == v) & (assigned == k)).sum()) for k in range(4)}
            for g, v in (("GO", 1), ("NOGO", 0))
        },
        "template_to_template_distance": {
            TYPE_NAMES[i]: {TYPE_NAMES[j]: float(((data_template[:, i] - data_template[:, j]) ** 2).mean()) for j in range(4)}
            for i in range(4)
        },
    }

    # ---- D: figure ------------------------------------------------------------------
    fd = model.filter_fun1(train_spikes)
    fm = model.filter_fun1(model_spikes)
    dstim_f = dstim
    tt = np.linspace(opt.start, opt.stop, fd.shape[0])
    fig, axes = plt.subplots(1, num_areas, figsize=(5.5 * num_areas, 3.8), squeeze=False)
    for a_i, aname in enumerate(opt.areas):
        ax = axes[0, a_i]
        na = (area == a_i).to(fd.device)
        for g, gname, col in ((1, "GO", "tab:blue"), (0, "NO-GO", "tab:orange")):
            d = torch.where((dstim_f == g)[None], fd, torch.full_like(fd, float("nan")))
            d = torch.nanmean(d, dim=1)[:, na].mean(1).cpu().numpy() / ts
            m = fm[:, m_stim == g][:, :, na].mean((1, 2)).cpu().numpy() / ts
            ax.plot(tt, d, color=col, label=f"data {gname}")
            ax.plot(tt, m, color=col, ls="--", label=f"model {gname}")
        for x in (0, 2, 4):
            ax.axvline(x, color="k", lw=0.5)
        ax.set_title(aname)
        ax.set_xlabel("time from sound onset (s)")
        ax.set_ylabel("population mean rate (Hz)")
        if a_i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "stimulus_check.png"), dpi=130)

    json.dump(summary, open(os.path.join(args.out, "stimulus_check.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
