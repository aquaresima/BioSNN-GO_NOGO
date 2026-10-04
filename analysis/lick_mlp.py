"""Lick-count readout (as in the paper's behaviour classifier, but on firing rates and for lick counts).

Per session: an MLP with two hidden layers of 128 units (as in Sourmpis et al.) maps the firing rate of each cell
(delay 2-4 s and early response 4-4.7 s, the part of the trial the model simulates) to the number of licks in a window,
with a Poisson loss. It is trained on REAL activity and trials, scored on held-out real trials, and then applied
unchanged to the activity of the same neurons in the MODEL for simulated GO and NO-GO trials. The readout is never
used in training the model.

Targets: "early" = licks in 4.0-4.7 s (during the simulated window), "late" = licks in 4.7-6 s (after it).

Usage: python -m analysis.lick_mlp --run log_dir/master/<run> --out analysis/out/<name> [--ckpt last]
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from analysis.common import load_checkpoint_and_data, load_dataset_with_trials, simulate
from analysis.licks import session_lick_counts
from infopath.utils.eval_stats import _window

DELAY, RESP = (2.0, 4.0), (4.0, 4.7)


def features(spikes, idx, opt, ts, n_trials=None):
    T = spikes.shape[0]
    idx = idx.to(spikes.device)
    cols = []
    for t0, t1 in (DELAY, RESP):
        a, b = _window(opt, t0, t1, T)
        x = spikes[a:b][:, :n_trials][:, :, idx].float().mean(0) / ts
        cols.append(torch.log1p(x))
    out = torch.cat(cols, 1).cpu().numpy()
    assert not np.isnan(out).any(), "NaN in features"
    return out


class MLP(torch.nn.Module):
    def __init__(self, d, h=128, p=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(p),
            torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Dropout(p),
            torch.nn.Linear(h, 1),
        )

    def forward(self, x):
        # log of the expected count; inputs are clipped to the range seen in training and the output is bounded to
        # [exp(-6), 40 licks] so that out-of-range activity (e.g. the model's) cannot give absurd predictions
        x = torch.clamp(x, -5.0, 5.0)
        return torch.clamp(self.net(x).squeeze(-1), -6.0, float(np.log(40.0)))


def fit_ensemble(Xtr, ytr, n_models=5, epochs=400, wd=5e-2, lr=1e-3, seed=0):
    """Poisson MLPs; early stopping on a 20% validation split of the training trials. Returns a predictor."""
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    loss_fn = torch.nn.PoissonNLLLoss(log_input=True)
    models = []
    for m in range(n_models):
        g = torch.Generator().manual_seed(seed + m)
        perm = torch.randperm(len(ytr_t), generator=g)
        nv = max(int(0.2 * len(perm)), 8)
        va, tr = perm[:nv], perm[nv:]
        torch.manual_seed(seed + m)
        net = MLP(Xtr.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        best, best_state, patience = 1e9, None, 0
        for ep in range(epochs):
            net.train()
            opt.zero_grad()
            loss_fn(net(Xtr_t[tr]), ytr_t[tr]).backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xtr_t[va]), ytr_t[va]).item()
            if v < best - 1e-4:
                best, best_state, patience = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 40:
                    break
        net.load_state_dict(best_state)
        net.eval()
        models.append(net)

    def predict(X):
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            return np.exp(np.mean([m(Xt).numpy() for m in models], axis=0))  # geometric-mean ensemble

    return predict


def poisson_deviance(y, mu):
    mu = np.clip(mu, 1e-6, None)
    y_log = np.where(y > 0, y * np.log(np.clip(y, 1e-12, None) / mu), 0.0)
    return float(2 * np.sum(y_log - (y - mu)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--min_neurons", type=int, default=10)
    args = ap.parse_args()

    opt, model, _, _, _, _ = load_checkpoint_and_data(args.run, args.out, args.ckpt, args.n_trials)
    ts = model.timestep
    train_sp, info_tr, test_sp, info_te, sessions, train_idx, test_idx = load_dataset_with_trials(model)
    model_spikes, m_stim = simulate(model, opt, args.n_trials)
    m_stim = m_stim.cpu().numpy()
    area = model.rsnn.area_index.cpu()
    results = []
    for s, name in enumerate(sessions):
        idx = torch.as_tensor(info_tr[-1][s]).bool()
        if int(idx.sum()) < args.min_neurons:
            continue
        counts = session_lick_counts(os.path.join(opt.datapath, name))
        st = np.asarray(info_tr[1][s]); ste = np.asarray(info_te[1][s])
        assert len(train_idx[s]) == len(st) and len(test_idx[s]) == len(ste), "train/test index mismatch"
        Xtr = features(train_sp, idx, opt, ts, len(st))
        Xte = features(test_sp, idx, opt, ts, len(ste))
        Xm = features(model_spikes, idx, opt, ts)
        mu, sd = Xtr.mean(0), np.clip(Xtr.std(0), 1e-3, None)
        Xtr, Xte, Xm = (Xtr - mu) / sd, (Xte - mu) / sd, (Xm - mu) / sd
        row = {"session": name, "area": opt.areas[int(area[idx.cpu()].mode().values)], "n_neurons": int(idx.sum()),
               "n_train": len(st), "n_test": len(ste)}
        for j, tname in ((0, "early"), (1, "late")):
            ytr, yte = counts[train_idx[s], j].astype(float), counts[test_idx[s], j].astype(float)
            pred = fit_ensemble(Xtr, ytr)
            p_te, p_m = pred(Xte), pred(Xm)
            # baselines on the same held-out trials
            mu_null = np.full_like(yte, ytr.mean())
            mu_stim = np.where(ste == 1, ytr[st == 1].mean(), ytr[st == 0].mean())
            D0, Ds, Dm = (poisson_deviance(yte, m) for m in (mu_null, mu_stim, p_te))
            rng = np.random.default_rng(0)
            rho = float(spearmanr(p_te, yte).statistic)
            null = [spearmanr(rng.permutation(p_te), yte).statistic for _ in range(1000)]
            within = {}
            for g, gname in ((1, "GO"), (0, "NOGO")):
                mk = ste == g
                if mk.sum() > 10 and yte[mk].std() > 0:
                    within[gname] = float(spearmanr(p_te[mk], yte[mk]).statistic)
            row[tname] = {
                "obs_mean_GO": float(yte[ste == 1].mean()), "obs_mean_NOGO": float(yte[ste == 0].mean()),
                "pred_mean_GO_data": float(p_te[ste == 1].mean()), "pred_mean_NOGO_data": float(p_te[ste == 0].mean()),
                "spearman_heldout": rho, "spearman_null_p95": float(np.percentile(null, 95)),
                "spearman_within_GO": within.get("GO"), "spearman_within_NOGO": within.get("NOGO"),
                "deviance_explained_vs_mean": 1 - Dm / D0, "deviance_explained_stimulus_only": 1 - Ds / D0,
                "model_pred_mean_GO": float(p_m[m_stim == 1].mean()), "model_pred_mean_NOGO": float(p_m[m_stim == 0].mean()),
                "model_pred_sd_GO": float(p_m[m_stim == 1].std()), "model_pred_sd_NOGO": float(p_m[m_stim == 0].std()),
            }
        results.append(row)
        print(name[:24], row["area"], "n", row["n_neurons"], flush=True)

    json.dump({"run": args.run, "ckpt": args.ckpt, "sessions": results}, open(os.path.join(args.out, "lick_mlp.json"), "w"), indent=1)
    print("\nsession area  n | target | observed lick count GO/NOGO | predicted (real data) GO/NOGO | held-out rho (null 95th) within-GO/within-NOGO | dev. explained: MLP vs stimulus-only | MODEL predicted GO/NOGO")
    for r in results:
        for t in ("early", "late"):
            d = r[t]
            w = lambda v: "  na" if v is None else f"{v:+.2f}"
            print(f"{r['session'][:12]:12s} {r['area']:4s} {r['n_neurons']:3d} | {t:5s} | {d['obs_mean_GO']:.1f}/{d['obs_mean_NOGO']:.1f} | {d['pred_mean_GO_data']:.1f}/{d['pred_mean_NOGO_data']:.1f} | "
                  f"{d['spearman_heldout']:+.2f} ({d['spearman_null_p95']:.2f}) {w(d['spearman_within_GO'])}/{w(d['spearman_within_NOGO'])} | "
                  f"{d['deviance_explained_vs_mean']:+.2f} vs {d['deviance_explained_stimulus_only']:+.2f} | {d['model_pred_mean_GO']:.1f}/{d['model_pred_mean_NOGO']:.1f}")


if __name__ == "__main__":
    main()
