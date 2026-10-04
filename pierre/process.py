"""
Convert AC/ALM NWB sessions into the folder structure expected by datasets/dataloader.py::TrialDataset 
and infopath/session_stitching.py::build_network (same layout as datasets/datastructure2datasetandvideo_Vahid.py).
"""

import os
import h5py
import numpy as np
import pandas as pd

# EXC/INH: Reuses the convention from datasets/datastructure2datasetandvideo_Vahid.py:
#     excitatory = peak_to_trough_width > WIDTH_THR
# Originally, `mat["width"]` (ms). We use `peak_to_valley`. 

WIDTH_THR_S = 0.275e-3  # 0.275 ms
# Folder with the raw .nwb sessions (not part of this repository) and the folder the converted dataset is written to.
INPUT_FOLDER = os.environ.get("PIERRE_NWB", "./data/nwb")
OUTPUT_FOLDER = os.environ.get("PIERRE_OUT", "./datasets/Pierre_v2")
SOUND_DURATION_S = 2.0

def classify_excitatory(peak_to_valley_seconds):
    return np.asarray(peak_to_valley_seconds) > WIDTH_THR_S

TRIAL_TYPE_MAP = {"Hit": "Hit", "Miss": "Miss", "Correct": "CR", "FalseAlarm": "FA"}

def _decode(x):
    if isinstance(x, bytes): 
        return x.decode('utf-8')
    return str(x)


def sound_onsets_npx(ttl, n_trials, tol=0.05):
    """Sound onsets on the neural (NPX) clock.

    `TTLreceivedNPX` holds two events per trial: sound onset and sound offset (SOUND_DURATION_S apart).
    Pair each onset with the event one sound-duration later and keep the onsets. Duplicated
    timestamps (seen in one session) never form a pair and are skipped. The number of pairs must
    equal the number of trials.
    """
    onsets, i = [], 0
    while i < len(ttl) - 1:
        if abs(ttl[i + 1] - ttl[i] - SOUND_DURATION_S) < tol:
            onsets.append(ttl[i])
            i += 2
        else:
            i += 1
    if len(onsets) != n_trials:
        raise ValueError(f"found {len(onsets)} sound onsets for {n_trials} trials")
    return np.array(onsets)


def sound_onsets_behaviour(io, n_trials, fs=1000.0, block_s=None):
    """Sound onsets on the behaviour clock shared by Licks and SoundCopy (1 kHz, concatenated trial blocks).

    Each trial block starts at Trials.start_time; the onset is the first sample where |SoundCopy|
    exceeds 20% of the block maximum.
    """
    acq = io["acquisition"]
    snd = np.abs(acq["SoundCopy"]["data"][:].astype(float))
    starts = acq["Trials"]["start_time"][:]
    block = int(round(np.median(np.diff(starts)) * fs)) if block_s is None else int(block_s * fs)
    onsets = np.full(n_trials, np.nan)
    for t in range(n_trials):
        a = int(round(starts[t] * fs))
        seg = snd[a : a + block]
        if len(seg) and seg.max() > 0:
            onsets[t] = starts[t] + np.flatnonzero(seg > 0.2 * seg.max())[0] / fs
    if np.isnan(onsets).any():
        raise ValueError(f"{np.isnan(onsets).sum()} trials without a detectable sound in SoundCopy")
    return onsets

def convert_session(in_folder=INPUT_FOLDER, out_folder=OUTPUT_FOLDER, lick_window=(-0.1, 7.0), lick_fs=1000.0):
    """Convert a single .nwb file into the target folder structure. Returns session_name."""
    for filename in sorted(os.listdir(in_folder)):
        if not filename.endswith(".nwb"):
            continue

        path = os.path.join(in_folder, filename)
        session_name = os.path.splitext(filename)[0]
        print(f"Processing: {filename}...")

        try:
            io = h5py.File(path, "r")
        except OSError:
            print(f"ERROR: '{filename}'. Corrupt or not valid. Skipping...")
            continue

        with io:
            session_path = os.path.join(out_folder, session_name)
            os.makedirs(session_path, exist_ok=True)
            os.makedirs(os.path.join(session_path, "jaw_trace"), exist_ok=True)

            if "_ALM_" in path:
                area = "ALM"
            elif "_AC_" in path:
                area = "AC"
            else:
                area = "unknown"

            # TIME ALIGNMENT
            zero_t = 0.0
            if "processing" in io and "Alignment_info" in io["processing"] and "Zero_t" in io["processing"]["Alignment_info"]:
                zero_t = io["processing"]["Alignment_info"]["Zero_t"]["data"][0]
                print(f"  -> zero_t detected: {zero_t:.4f}s")
            else:
                print("  -> zero_t not found. Assuming 0.0s.")

            units = io["units"]
            n_units = len(units["id"])  

            ptv_col = units["peak_to_valley"][:] if "peak_to_valley" in units else np.full(n_units, np.nan)
            cluster_id_col = units["ks_unit_id"][:] if "ks_unit_id" in units else np.arange(n_units)
            depth = units["depth"][:] if "depth" in units else np.full(n_units, np.nan)
            firing_rate = units["firing_rate"][:] if "firing_rate" in units else np.full(n_units, np.nan)

            cluster_df = pd.DataFrame({
                    "neuron_index": np.arange(n_units),
                    "area": area,
                    "excitatory": classify_excitatory(ptv_col),  # Corregido: pasar directamente el array
                    "depth": depth,
                    "cluster": cluster_id_col,
                    "firing_rate": firing_rate,
                    "with_video": 0,  
            })
            cluster_df.to_csv(os.path.join(session_path, "cluster_info"))
            n_exc = cluster_df['excitatory'].sum()
            perc_exc = (n_exc / n_units) * 100 if n_units > 0 else 0
            perc_inh = 100.0 - perc_exc
            print(f"  -> Neuronas: {n_units} totales | {perc_exc:.1f}% Exc | {perc_inh:.1f}% Inh")
            
            spike_times_flat = units["spike_times"]
            spike_times_idx = units["spike_times_index"][:]
            start_idx = 0
            for i in range(n_units):
                end_idx = spike_times_idx[i]
                spike_times = spike_times_flat[start_idx:end_idx] - zero_t
                np.save(os.path.join(session_path, f"neuron_index_{i}"), spike_times)
                start_idx = end_idx

            # Trials
            trials_df = io["acquisition"]["Trials"]
            n_trials = len(trials_df["id"]) 

            trial_type_raw = [_decode(t) for t in trials_df["HMCF"][:]]    
            trial_type = [TRIAL_TYPE_MAP.get(t, "Miss") for t in trial_type_raw]
            
            response_times = trials_df["response_time"][:] if "response_time" in trials_df else np.full(n_trials, np.nan)

            go_nogo_raw = [_decode(t) for t in trials_df["trial_type"][:]]
            stim = np.array([1 if g=="Go" else 0 for g in go_nogo_raw], dtype=int)

            ttl_npx = io["acquisition"]["TTLreceivedNPX"]["data"][:]
            sound_onsets = sound_onsets_npx(ttl_npx, n_trials) - zero_t
            lick_onsets = sound_onsets_behaviour(io, n_trials, fs=lick_fs)

            trial_info = pd.DataFrame({
                    "trial_number": np.arange(n_trials),
                    "reaction_time_piezo": response_times,
                    "reaction_time_jaw": response_times, 
                    "stim": stim, 
                    "trial_active": np.ones(n_trials, dtype=int),
                    "trial_type": trial_type,
                    "trial_onset": sound_onsets,
                    "jaw_trace": [
                        os.path.normpath(os.path.join(session_path, "jaw_trace", f"trial_{t}"))
                        for t in range(n_trials)
                    ],
                    "tongue_trace": "",
                    "whisker_angle": "",
                    "completed_trials": np.ones(n_trials, dtype=int),
                    "video_onset": -1,
                    "video_offset": 3,
                })
            trial_info.to_csv(os.path.join(session_path, "trial_info"))

            # Licks
            licks = io["acquisition"]["Licks"]["data"][:]
            print(f"  -> Diagnóstico: {n_trials} trials listos | {len(licks)} licks detectados")
            save_lick_traces(licks, lick_onsets, session_path, window=lick_window, fs=lick_fs)

    return session_name


def save_lick_traces(lick_data, trial_onsets, session_path, window=(-0.1, 3.0), fs=1000.0):
    for t, onset in enumerate(trial_onsets):
        start_sec = onset + window[0]
        end_sec = onset + window[1]

        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)

        if start_idx < 0:
            pad_length = -start_idx
            slice_data = lick_data[0:end_idx]
            slice_data = np.pad(slice_data, (pad_length, 0), mode='edge')
        elif end_idx > len(lick_data):
            pad_length = end_idx - len(lick_data)
            slice_data = lick_data[start_idx:len(lick_data)]
            slice_data = np.pad(slice_data, (0, pad_length), mode='edge')
        else:
            slice_data = lick_data[start_idx:end_idx]

        #trace = gaussian_filter1d(slice_data.astype(float), sigma=4)
        np.save(os.path.join(session_path, "jaw_trace", f"trial_{t}"), slice_data)

def unify_cluster_table(out_folder=OUTPUT_FOLDER):
    sessions = [d for d in os.listdir(out_folder) if os.path.isdir(os.path.join(out_folder, d))]
    all_df = []
    for sess in sessions:
        df = pd.read_csv(os.path.join(out_folder, sess, "cluster_info"), index_col=0)
        df = df.assign(session=sess, cluster_index=df.index.values)
        all_df.append(df[["session", "area", "excitatory", "firing_rate", "cluster_index", "with_video"]])
    pd.concat(all_df, ignore_index=True).to_csv(os.path.join(out_folder, "cluster_information"))


if __name__ == "__main__":
    convert_session()
    unify_cluster_table()
    print("Data converted successfully.")
