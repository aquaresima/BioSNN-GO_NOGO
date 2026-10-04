import torch
import torch.nn as nn
import numpy as np


class InputSpikes(nn.Module):
    def __init__(self, opt, prec_type=torch.float32):
        super(InputSpikes, self).__init__()
        self.prec_type = prec_type
        self.opt = opt
        if opt.n_rnn_in == 200:
            p_stim = 0.5
        elif opt.n_rnn_in == 300:
            p_stim = 2 / 3
        else:
            p_stim = 1
        self.p_stim = p_stim

    def forward(self, stim):
        dt_in_seconds = self.opt.dt / 1000.0
        n_neurons = self.opt.n_rnn_in
        start = -int(self.opt.start / dt_in_seconds)
        stop = int(self.opt.stop / dt_in_seconds)
        thalamic_delay = int(self.opt.thalamic_delay / dt_in_seconds)
        stim_dur = int(self.opt.stim_duration / dt_in_seconds)
        firing_prob = self.opt.input_f0 * dt_in_seconds
        batch_size = stim.shape[0]
        pattern = (torch.ones((start + stop, batch_size, n_neurons), device=stim.device)* firing_prob)
        pattern[pattern > 1] = 1
        n_stim = int(n_neurons * self.p_stim / len(self.opt.stim_onsets))
        for i, j in enumerate(stim):
            for l, k in enumerate(self.opt.stim_onsets):
                start = int(np.round((k - self.opt.start) / dt_in_seconds))
                scale = (
                    self.opt.scale_fun(j.item()) if k == 0 else self.opt.scale_fun(1)
                )
                pattern[
                    start + thalamic_delay : start + thalamic_delay + stim_dur,
                    i,
                    n_stim * l : n_stim * (l + 1),
                ] *= (
                    1 + self.opt.stim_valance * scale
                )
        pattern[pattern > 1] = 1
        if n_neurons == len(self.opt.stim_onsets):
            return (pattern > firing_prob) * 1.0 * self.opt.dt / 2
        return torch.bernoulli(pattern)


class InputSpikes_Adapted(nn.Module):
    """
    Input CONTINUO (no espikeado): en vez de muestrear Bernoulli sobre una
    probabilidad de disparo, se entrega directamente la envolvente analógica
    (population-coded en 3 canales: constante / GO / NOGO) como corriente de
    entrada al RSNN. Sin sampling estocástico -> sin ruido añadido encima del
    decay continuo que ya define tau_sound_decay, y sin varianza espuria entre
    trials idénticos. La escala *dt/2 replica la convención que ya usa
    InputSpikes en su rama determinista (ver arriba).
    """

    def __init__(self, opt, prec_type=torch.float32):
        super().__init__()
        self.prec_type = prec_type
        self.opt = opt
        self.tau_sound_decay = getattr(opt, "tau_sound_decay", 0.5)
        # optional response-window cue (lick port arrives): one extra channel, same for GO and NO-GO trials
        self.n_cue = 1 if getattr(opt, "cue_input", False) else 0
        assert opt.n_rnn_in >= 3 + self.n_cue, "InputSpikes_Adapted necesita 3 canales de sonido (+1 si cue_input)"

    def forward(self, stim):
        opt = self.opt
        # stim debe ser binario (0=NOGO, 1=GO); si tienes más tipos de trial hay que revisar esto
        assert set(torch.unique(stim).tolist()).issubset({0.0, 1.0}), \
            "InputSpikes_Adapted solo soporta estimulación GO/NOGO binaria"

        dt_in_seconds = opt.dt / 1000.0
        n_neurons = opt.n_rnn_in

        start = -int(opt.start / dt_in_seconds)
        stop = int(opt.stop / dt_in_seconds)
        sound_dur = int(opt.stim_duration / dt_in_seconds)
        thalamic_delay = int(opt.thalamic_delay / dt_in_seconds)
        tau_bins = self.tau_sound_decay / dt_in_seconds
        firing_prob = opt.input_f0 * dt_in_seconds
        batch_size = stim.shape[0]

        # divide las n_rnn_in neuronas en 3 poblaciones: constante / GO / NOGO
        n_sound = n_neurons - self.n_cue
        n_per_pop = n_sound // 3
        pop_const = slice(0, n_per_pop)
        pop_go = slice(n_per_pop, 2 * n_per_pop)
        pop_nogo = slice(2 * n_per_pop, n_sound)  # se queda con el resto si no divide exacto
        pop_cue = slice(n_sound, n_neurons)

        pattern = torch.ones((start + stop, batch_size, n_neurons), device=stim.device) * firing_prob
        pattern.clamp_(max=1)

        k = opt.stim_onsets[0]
        onset = int(np.round((k - opt.start) / dt_in_seconds))
        t0 = onset + thalamic_delay
        t1 = min(t0 + sound_dur, pattern.shape[0])
        dur = t1 - t0
        if dur <= 0:
            # sin ventana de estímulo válida: solo el nivel base continuo, sin spikes
            return pattern * opt.dt / 2

        decay = torch.exp(-torch.arange(dur, device=stim.device, dtype=self.prec_type) / tau_bins)
        scale = opt.scale_fun(1)
        valance = opt.stim_valance * scale
        is_go = (stim > 0.5)

        # canal constante: mismo perfil en todos los trials, GO o NOGO
        pattern[t0:t1, :, pop_const] *= (1 + valance)

        decay_col = decay[:, None, None]          # (dur, 1, 1)
        go_mask = is_go[None, :, None].float()     # (1, batch, 1)
        nogo_mask = 1 - go_mask

        # población GO solo se modula en trials GO, población NOGO solo en trials NOGO
        pattern[t0:t1, :, pop_go] *= (1 + valance * decay_col) * go_mask + (1 - go_mask)
        pattern[t0:t1, :, pop_nogo] *= (1 + valance * decay_col) * nogo_mask + (1 - nogo_mask)

        if self.n_cue:
            # rectangular pulse of cue_duration at cue_time (relative to sound onset), same drive as the sound
            # channels; skipped when the simulated window does not reach it (pre-stimulus copy)
            c0 = int(np.round((opt.cue_time - opt.start) / dt_in_seconds)) + thalamic_delay
            c1 = min(c0 + int(np.round(opt.cue_duration / dt_in_seconds)), pattern.shape[0])
            if 0 <= c0 < c1:
                pattern[c0:c1, :, pop_cue] *= 1 + valance

        pattern.clamp_(max=1)

        # SIN Bernoulli: se devuelve la envolvente continua directamente como
        # corriente de entrada, en vez de espikearla.
        return pattern * opt.dt / 2
