import torch
import time
import json
import os


class Timer:
    def __init__(self, device=None):
        self.device = device
        self.dt = None
        self._t0 = None

    def __enter__(self):
        if self.device is not None and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device is not None and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self.dt = time.time() - self._t0
        return False  # nunca suprime excepciones


def count_trainable_parameters(model, verbose=True):
    breakdown = {}
    total_trainable, total_frozen = 0, 0
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        n = p.numel()
        breakdown.setdefault(top, {"trainable": 0, "frozen": 0, "names": []})
        if p.requires_grad:
            total_trainable += n
            breakdown[top]["trainable"] += n
        else:
            total_frozen += n
            breakdown[top]["frozen"] += n
        breakdown[top]["names"].append((name, n, p.requires_grad))

    if verbose:
        print(f"Total trainable parameters: {total_trainable:,}")
        print(f"Total frozen parameters (requires_grad=False): {total_frozen:,}")
        for k, v in breakdown.items():
            print(f"  {k:20s} trainable={v['trainable']:>10,}  frozen={v['frozen']:>10,}")

    compact = {k: {"trainable": v["trainable"], "frozen": v["frozen"]} for k, v in breakdown.items()}
    return {"total_trainable": total_trainable, "total_frozen": total_frozen, "breakdown": compact}


@torch.no_grad()
def count_active_weights(model):
    rsnn = model.rsnn
    stats = {}

    def _frac(name, w):
        active = int((w.abs() > 1e-7).sum().item())
        total = int(w.numel())
        stats[f"{name}_active"] = active
        stats[f"{name}_total"] = total
        stats[f"{name}_active_frac"] = active / total if total > 0 else 0.0

    _frac("w_rec", rsnn._w_rec.data)
    _frac("w_in", rsnn._w_in.data)
    if rsnn.motor_areas.shape[0] > 0:
        _frac("w_jaw_pre", rsnn._w_jaw_pre.data)
    return stats


@torch.no_grad()
def neuron_activity_stats(spikes, model, timestep=None, prefix=""):
    spikes = spikes.detach()
    T, K, N = spikes.shape
    timestep = timestep or model.timestep

    active = spikes > 0                                        # (T, K, N)
    last_bin_active = active[-1].any(dim=0)                     # (N,)
    before_active = (
        active[:-1].any(dim=0).any(dim=0)
        if T > 1
        else torch.zeros(N, dtype=torch.bool, device=spikes.device)
    )
    ever_active = last_bin_active | before_active

    fr = spikes.mean(dim=(0, 1)) / timestep  # Hz, por neurona (N,)

    exc_idx = model.rsnn.excitatory_index
    area_idx = model.rsnn.area_index

    stats = {
        f"{prefix}frac_active_last_bin": last_bin_active.float().mean().item(),
        f"{prefix}frac_active_before_last_bin": before_active.float().mean().item(),
        f"{prefix}frac_neurons_ever_active": ever_active.float().mean().item(),
        f"{prefix}frac_neurons_dead": (~ever_active).float().mean().item(),
        f"{prefix}mean_firing_rate_hz": fr.mean().item(),
        f"{prefix}std_firing_rate_hz": fr.std().item(),
        f"{prefix}max_firing_rate_hz": fr.max().item(),
        f"{prefix}min_firing_rate_hz": fr.min().item(),
        f"{prefix}mean_fr_exc_hz": fr[exc_idx].mean().item(),
        f"{prefix}mean_fr_inh_hz": fr[~exc_idx].mean().item(),
    }

    for a in range(model.num_areas):
        area_name = model.opt.areas[a]
        mask = area_idx == a
        if mask.any():
            stats[f"{prefix}mean_fr_{area_name}_hz"] = fr[mask].mean().item()
            stats[f"{prefix}frac_active_{area_name}"] = ever_active[mask].float().mean().item()

    return stats


class DiagnosticsLogger:
    def __init__(self, log_path, filename="diagnostics.jsonl", step_every=50, flush_every=1):
        self.path = os.path.join(log_path, filename)
        self._file = open(self.path, "a")
        self._start_time = time.time()
        self._last_time = self._start_time
        self.step_every = step_every
        self.flush_every = flush_every
        self._counter = 0
        self._n_since_flush = 0

    def log(self, step, model=None, model_spikes=None, extra=None, log_params=False, force_full=False):
        now = time.time()
        entry = {
            "step": step,
            "wall_time": now,
            "iso_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "dt_since_last_log": now - self._last_time,
            "elapsed_since_start": now - self._start_time,
        }
        self._last_time = now
        self._counter += 1

        do_full = force_full or (self._counter % self.step_every == 0)

        if do_full and model_spikes is not None and model is not None:
            entry.update(neuron_activity_stats(model_spikes, model))
            entry.update(count_active_weights(model))

        if log_params and model is not None:
            entry["param_counts"] = count_trainable_parameters(model, verbose=False)

        if extra is not None:
            entry.update({k: v for k, v in extra.items() if v is not None})

        self._file.write(json.dumps(entry) + "\n")
        self._n_since_flush += 1
        if self._n_since_flush >= self.flush_every:
            self._file.flush()
            self._n_since_flush = 0

    def close(self):
        self._file.close()


def load_diagnostics(log_path, filename="diagnostics.jsonl"):
    import pandas as pd
    return pd.read_json(os.path.join(log_path, filename), lines=True)
