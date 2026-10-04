"""Behaviour test: does the model's spiking activity carry the lick / no-lick mapping of the real neurons?

For every session, a logistic-regression readout is fit on the REAL activity of the neurons that are in the model
(lick = Hit or False Alarm, no lick = Miss or Correct Rejection) from firing rates in the delay (2-4 s) and/or the
first part of the response window (4 s - stop). It is scored on held-out real trials. The same fixed readout is then
applied to the MODEL activity of the same neurons for simulated GO and NO-GO trials. If the model only matches
firing-rate statistics, the decoded lick probability is the same for GO and NO-GO; if it has learned the mapping, it is
high for GO and low for NO-GO, as in the data.

The readout is never used in training. Usage:
  python analysis/lick_decoder_test.py --run log_dir/master/<run> --out analysis/out/<name> [--ckpt last]
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from analysis.common import load_checkpoint_and_data, simulate
from infopath.utils.eval_stats import _window

FEATURES = {"delay": [(2.0, 4.0)], "response": [(4.0, None)], "delay+response": [(2.0, 4.0), (4.0, None)]}


def rate_features(spikes, idx, windows, opt, timestep, n_trials=None):
    """log(1 + rate in Hz) per neuron and window. spikes (T, trials, N); idx boolean mask of the session's neurons."""
    T = spikes.shape[0]
    idx = idx.to(spikes.device)
    cols = []
    for t0, t1 in windows:
        a, b = _window(opt, t0, t1 if t1 is not None else opt.stop, T)
        x = spikes[a:b][:, :n_trials][:, :, idx].float().mean(0) / timestep  # (trials, n)
        cols.append(torch.log1p(x))
    out = torch.cat(cols, dim=1).cpu().numpy()
    assert not np.isnan(out).any(), "NaN in features: session trial layout does not match session_info"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--min_neurons", type=int, default=10)
    ap.add_argument("--n_perm", type=int, default=50)
    args = ap.parse_args()

    opt, model, train_spikes, info_train, test_spikes, info_test = load_checkpoint_and_data(
        args.run, args.out, args.ckpt, args.n_trials
    )
    ts = model.timestep
    model_spikes, m_stim = simulate(model, opt, args.n_trials)
    m_stim = m_stim.cpu().numpy()
    area = model.rsnn.area_index.cpu()
    sessions = []
    for s, idx in enumerate(info_train[-1]):
        idx = torch.as_tensor(idx).bool()
        n_neurons = int(idx.sum())
        a_i = int(area[idx.cpu()].mode().values)
        tt_tr, tt_te = np.asarray(info_train[0][s]), np.asarray(info_test[0][s])
        st_tr, st_te = np.asarray(info_train[1][s]), np.asarray(info_test[1][s])
        y_tr, y_te = np.isin(tt_tr, [1, 3]).astype(int), np.isin(tt_te, [1, 3]).astype(int)  # Hit, FA = lick
        row = {"session": s, "area": opt.areas[a_i], "n_neurons": n_neurons,
               "n_train": len(y_tr), "n_test": len(y_te),
               "data_lick_rate_GO": float(np.isin(np.r_[tt_tr, tt_te][np.r_[st_tr, st_te] == 1], [1, 3]).mean()),
               "data_lick_rate_NOGO": float(np.isin(np.r_[tt_tr, tt_te][np.r_[st_tr, st_te] == 0], [1, 3]).mean())}
        if n_neurons < args.min_neurons or len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            row["skipped"] = True
            sessions.append(row)
            continue
        for fname, windows in FEATURES.items():
            Xtr = rate_features(train_spikes, idx, windows, opt, ts, len(y_tr))
            Xte = rate_features(test_spikes, idx, windows, opt, ts, len(y_te))
            Xm = rate_features(model_spikes, idx.to(model_spikes.device), windows, opt, ts)
            mu, sd = Xtr.mean(0), np.clip(Xtr.std(0), 1e-3, None)
            clf = LogisticRegression(C=args.C, max_iter=2000).fit((Xtr - mu) / sd, y_tr)
            p_te = clf.predict_proba((Xte - mu) / sd)[:, 1]
            p_m = clf.predict_proba((Xm - mu) / sd)[:, 1]
            # null: same pipeline with shuffled training labels (chance level of the held-out AUC)
            rng = np.random.default_rng(0)
            null = []
            for _ in range(args.n_perm):
                c = LogisticRegression(C=args.C, max_iter=2000).fit((Xtr - mu) / sd, rng.permutation(y_tr))
                null.append(roc_auc_score(y_te, c.predict_proba((Xte - mu) / sd)[:, 1]))
            # null for the model: AUC of the decoded probability against shuffled stimulus labels
            mnull = [roc_auc_score(rng.permutation(m_stim), p_m) for _ in range(1000)]
            row[fname] = {
                "data_heldout_auc_null_mean": float(np.mean(null)),
                "data_heldout_auc_null_p95": float(np.percentile(null, 95)),
                "model_auc_null_p95": float(np.percentile(mnull, 95)),
                "model_auc_null_p05": float(np.percentile(mnull, 5)),
                "data_heldout_auc": float(roc_auc_score(y_te, p_te)),
                "data_heldout_acc": float(((p_te > 0.5) == y_te).mean()),
                "data_decoded_GO": float(p_te[st_te == 1].mean()),
                "data_decoded_NOGO": float(p_te[st_te == 0].mean()),
                "model_decoded_GO": float(p_m[m_stim == 1].mean()),
                "model_decoded_NOGO": float(p_m[m_stim == 0].mean()),
                "model_auc_vs_stimulus": float(roc_auc_score(m_stim, p_m)),
            }
        # does activity predict licking beyond the stimulus? FA vs CR among NO-GO trials, 5-fold CV on all trials
        all_spikes = [(train_spikes, len(y_tr)), (test_spikes, len(y_te))]
        X_all = np.concatenate([rate_features(sp, idx, FEATURES["delay+response"], opt, ts, n) for sp, n in all_spikes])
        y_all, st_all = np.r_[y_tr, y_te], np.r_[st_tr, st_te]
        ng = st_all == 0
        Xn, yn = X_all[ng], y_all[ng]
        if min(yn.sum(), (1 - yn).sum()) >= 8:
            Xn = (Xn - Xn.mean(0)) / np.clip(Xn.std(0), 1e-3, None)
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            pn = cross_val_predict(LogisticRegression(C=args.C, max_iter=2000), Xn, yn, cv=cv, method="predict_proba")[:, 1]
            rng = np.random.default_rng(1)
            nn = []
            for _ in range(args.n_perm):
                yp = rng.permutation(yn)
                nn.append(roc_auc_score(yp, cross_val_predict(LogisticRegression(C=args.C, max_iter=2000), Xn, yp, cv=cv, method="predict_proba")[:, 1]))
            row["within_nogo_FA_vs_CR"] = {"n_FA": int(yn.sum()), "n_CR": int((1 - yn).sum()), "cv_auc": float(roc_auc_score(yn, pn)),
                                           "null_mean": float(np.mean(nn)), "null_p95": float(np.percentile(nn, 95))}
        sessions.append(row)

    summary = {"run": args.run, "ckpt": args.ckpt, "C": args.C, "sessions": sessions, "by_area": {}}
    for aname in opt.areas:
        rows = [r for r in sessions if r["area"] == aname and not r.get("skipped")]
        if not rows:
            continue
        summary["by_area"][aname] = {"n_sessions": len(rows)}
        for fname in FEATURES:
            summary["by_area"][aname][fname] = {
                k: float(np.mean([r[fname][k] for r in rows])) for k in rows[0][fname]
            }
    json.dump(summary, open(os.path.join(args.out, "lick_decoder.json"), "w"), indent=1)

    print("session area  n | data lick rate GO/NOGO | decoder (delay+response): held-out AUC | data decoded GO/NOGO | model decoded GO/NOGO | model AUC vs stimulus")
    for r in sessions:
        if r.get("skipped"):
            print(f"{r['session']:3d} {r['area']:4s} {r['n_neurons']:4d} skipped")
            continue
        d = r["delay+response"]
        print(f"{r['session']:3d} {r['area']:4s} {r['n_neurons']:4d} | {r['data_lick_rate_GO']:.2f}/{r['data_lick_rate_NOGO']:.2f} | {d['data_heldout_auc']:.2f} | "
              f"{d['data_decoded_GO']:.2f}/{d['data_decoded_NOGO']:.2f} | {d['model_decoded_GO']:.2f}/{d['model_decoded_NOGO']:.2f} | {d['model_auc_vs_stimulus']:.2f}")
    print("\nChance controls per session (feature set delay+response)")
    for r in sessions:
        if r.get("skipped"):
            continue
        d = r["delay+response"]; w = r.get("within_nogo_FA_vs_CR")
        print(f"{r['area']:4s} held-out AUC {d['data_heldout_auc']:.2f} (shuffled-label null mean {d['data_heldout_auc_null_mean']:.2f}, 95th pct {d['data_heldout_auc_null_p95']:.2f}) | "
              f"model AUC vs stimulus {d['model_auc_vs_stimulus']:.2f} (shuffled-stimulus 5-95%: {d['model_auc_null_p05']:.2f}-{d['model_auc_null_p95']:.2f})")
        if w:
            print(f"     within NO-GO, FA vs CR (n={w['n_FA']}/{w['n_CR']}): CV AUC {w['cv_auc']:.2f} (null mean {w['null_mean']:.2f}, 95th pct {w['null_p95']:.2f})")
    print("\nBy area, mean over sessions (feature set: held-out AUC, data decoded GO/NOGO, model decoded GO/NOGO, model AUC vs stimulus)")
    for aname, v in summary["by_area"].items():
        for fname in FEATURES:
            d = v[fname]
            print(f"{aname:4s} {fname:15s} n={v['n_sessions']} AUC {d['data_heldout_auc']:.2f} | data {d['data_decoded_GO']:.2f}/{d['data_decoded_NOGO']:.2f} | "
                  f"model {d['model_decoded_GO']:.2f}/{d['model_decoded_NOGO']:.2f} | model AUC {d['model_auc_vs_stimulus']:.2f}")


if __name__ == "__main__":
    main()
