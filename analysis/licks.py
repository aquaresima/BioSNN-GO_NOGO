"""Lick counts from the spout sensor trace.

Trace (1 kHz, sample 0 = 0.1 s before sound onset) is a two-level signal: about 0.13 at rest and about 5.07 during
tongue contact. A lick is a rising edge through the midpoint.
"""
import os

import numpy as np

THRESHOLD = 2.5
T0 = -0.1  # time of sample 0 relative to sound onset (s)
FS = 1000.0
WINDOWS = {"early": (4.0, 4.7), "late": (4.7, 6.0), "full": (4.0, 6.0)}


def lick_times(trace):
    """Times (s from sound onset) of the rising edges."""
    x = np.asarray(trace, dtype=float)
    up = np.flatnonzero((x[1:] > THRESHOLD) & (x[:-1] <= THRESHOLD)) + 1
    return up / FS + T0


def session_lick_counts(session_dir, n_trials=None):
    """(n_trials, 3) counts in the early (4-4.7 s), late (4.7-6 s) and full (4-6 s) response windows, in trial_info order."""
    import pandas as pd

    ti = pd.read_csv(os.path.join(session_dir, "trial_info"))
    n = len(ti) if n_trials is None else n_trials
    out = np.zeros((n, len(WINDOWS)), dtype=int)
    for k in range(n):
        lt = lick_times(np.load(os.path.join(session_dir, "jaw_trace", f"trial_{int(ti.trial_number.values[k])}.npy")))
        for j, (a, b) in enumerate(WINDOWS.values()):
            out[k, j] = int(((lt >= a) & (lt < b)).sum())
    return out
