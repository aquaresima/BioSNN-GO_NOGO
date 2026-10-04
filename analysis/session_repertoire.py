"""Do the recorded sessions contain the same repertoire of single-neuron responses?

Per neuron: GO and NO-GO PSTH (0.25 s bins, -0.5 .. 4.75 s from sound onset) and GO vs NO-GO selectivity (d') in the
sound / delay / response epochs. Neurons of each area are clustered on PSTH shape; the cluster composition of every
session is compared with what random assignment of neurons to sessions would give (permutation null on the mean
pairwise Jensen-Shannon divergence). Usage: python -m analysis.session_repertoire --data <Pierre_v2> --out <dir>
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDGES = np.arange(-0.5, 4.76, 0.25)
EPOCHS = {"sound": (0.0, 2.0), "delay": (2.0, 4.0), "response": (4.0, 4.75)}


def neuron_features(spikes, onsets, stim):
    """Per-bin rates (Hz) for GO and NO-GO trials, and per-trial epoch counts."""
    idx = np.searchsorted(spikes, onsets[:, None] + EDGES[None, :])
    counts = np.diff(idx, axis=1) / np.diff(EDGES)[None, :]  # (trials, bins) in Hz
    psth = np.concatenate([counts[stim == 1].mean(0), counts[stim == 0].mean(0)])
    sel = {}
    for name, (t0, t1) in EPOCHS.items():
        a = np.searchsorted(spikes, onsets + t0)
        b = np.searchsorted(spikes, onsets + t1)
        c = (b - a) / (t1 - t0)
        g, n = c[stim == 1], c[stim == 0]
        sel[name] = float((g.mean() - n.mean()) / np.sqrt((g.var() + n.var()) / 2 + 1e-3))
    return psth, sel, float(counts.mean())


def mean_pairwise_js(labels, groups, k):
    comp = np.array([np.bincount(labels[groups == g], minlength=k) / max((groups == g).sum(), 1) for g in np.unique(groups)])
    d = [jensenshannon(comp[i], comp[j]) ** 2 for i in range(len(comp)) for j in range(i + 1, len(comp))]
    return float(np.mean(d)), comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n_perm", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cl = pd.read_csv(os.path.join(args.data, "cluster_information"))
    rows, feats = [], []
    for sess, g in cl.groupby("session"):
        ti = pd.read_csv(os.path.join(args.data, sess, "trial_info"))
        onsets, stim = ti.trial_onset.values, ti.stim.values
        for _, r in g.iterrows():
            sp = np.load(os.path.join(args.data, sess, f"neuron_index_{int(r.cluster_index)}.npy"))
            psth, sel, rate = neuron_features(np.sort(sp), onsets, stim)
            rows.append({"session": sess, "area": r.area, "excitatory": bool(r.excitatory), "rate": rate, **{f"dprime_{k}": v for k, v in sel.items()}})
            feats.append(psth)
        print(sess, len(g), flush=True)
    df = pd.DataFrame(rows)
    X = np.array(feats)
    summary = {"k": args.k, "areas": {}}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ai, area in enumerate(sorted(df.area.unique())):
        m = (df.area == area).values
        Z = X[m]
        Z = (Z - Z.mean(1, keepdims=True)) / np.clip(Z.std(1, keepdims=True), 1e-3, None)  # shape only
        lab = KMeans(args.k, n_init=10, random_state=0).fit_predict(Z)
        sess = df.session.values[m]
        obs, comp = mean_pairwise_js(lab, sess, args.k)
        rng = np.random.default_rng(0)
        null = np.array([mean_pairwise_js(lab, rng.permutation(sess), args.k)[0] for _ in range(args.n_perm)])
        sub = df[m]
        per_sess = sub.groupby("session").agg(n=("rate", "size"), frac_exc=("excitatory", "mean"), rate_median=("rate", "median"),
                                               sel_sound=("dprime_sound", lambda x: float((x.abs() > 0.5).mean())),
                                               sel_delay=("dprime_delay", lambda x: float((x.abs() > 0.5).mean())),
                                               sel_response=("dprime_response", lambda x: float((x.abs() > 0.5).mean())))
        summary["areas"][area] = {
            "n_neurons": int(m.sum()),
            "mean_pairwise_JS_observed": obs, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p_value": float((null >= obs).mean()),
            "per_session": per_sess.round(3).to_dict("index"),
            "cluster_sizes": np.bincount(lab, minlength=args.k).tolist(),
        }
        ax = axes[ai, 0]
        names = [s.split("_")[0] + "_" + s.split("_")[1][4:] for s in np.unique(sess)]
        bottom = np.zeros(len(comp))
        for c in range(args.k):
            ax.bar(names, comp[:, c], bottom=bottom, label=f"c{c}")
            bottom += comp[:, c]
        ax.set_title(f"{area}: cluster composition per session"); ax.tick_params(axis="x", rotation=60)
        axes[ai, 1].boxplot([np.log10(sub.rate[sub.session == s] + 0.01) for s in np.unique(sess)], labels=names)
        axes[ai, 1].set_title(f"{area}: log10 mean rate"); axes[ai, 1].tick_params(axis="x", rotation=60)
        axes[ai, 2].boxplot([sub.dprime_delay[sub.session == s] for s in np.unique(sess)], labels=names)
        axes[ai, 2].set_title(f"{area}: GO vs NO-GO d' (delay)"); axes[ai, 2].tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "session_repertoire.png"), dpi=110)
    json.dump(summary, open(os.path.join(args.out, "session_repertoire.json"), "w"), indent=1)
    for area, v in summary["areas"].items():
        print(f"{area}: mean pairwise JS {v['mean_pairwise_JS_observed']:.4f} (random-assignment null mean {v['null_mean']:.4f}, 95th pct {v['null_p95']:.4f}, p={v['p_value']:.3f})")
        print(pd.DataFrame(v["per_session"]).T.to_string())


if __name__ == "__main__":
    main()
