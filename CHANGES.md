# Changes with respect to the original project

Reference: the original code is [Sourmpis/BiologicallyInformed](https://github.com/Sourmpis/BiologicallyInformed) at commit `e6f5984`
(Sourmpis, Petersen, Gerstner & Bellec, bioRxiv 2024). Everything not listed here is unchanged: the spiking network core (`models/`), the
Sinkhorn trial-matching and neuron-wise loss functions in `infopath/losses.py`, the loss splitter, the data loader (`datasets/dataloader.py`),
the figure notebooks, the pretrained models (`AllModels/`) and the synthetic-data configurations.

Origin of each change: **[A]** first adaptation to the GO/NO-GO data (internship of D. Carrillos, Institut Pasteur / Institut de l'Audition);
**[B]** this repository.

## 1. Data and task

- **[A, corrected in B] New NWB converter `pierre/process.py`** for a delayed auditory GO/NO-GO task (spike times, trial table with Hit/Miss/CR/FA
  outcomes, lick sensor, sound copy; excitatory/inhibitory from the spike peak-to-valley time). The original reads the whisker-detection dataset
  of Esmaeili et al. 2021 (`datasets/datastructure2datasetandvideo_Vahid.py`, unchanged and still available).
  Two corrections in B, both of which matter:
  - *Sound onsets.* The TTL array has two events per trial (sound onset and offset, 2.0 s apart). Onsets are now the first event of each pair
    (`sound_onsets_npx`, which fails if the number of onsets differs from the number of trials). The first version used the first `n_trials` events, which
    mixes onsets and offsets and misaligns every trial after the first.
  - *Lick traces.* The lick sensor and the sound copy share a continuous 1 kHz clock of concatenated trial blocks, unrelated to the neural clock.
    Traces are now cut around the sound onset found in the sound copy (`sound_onsets_behaviour`).
- **[B] Time window and onset.** `trial_onset = 0` (the original dataset stores the stimulus 1 s into the trial, the converted data store the sound
  onset itself) and a window of -0.2 to 4.7 s: sound (0-2 s), delay (2-4 s) and the first 0.7 s of the response window.
- **[B] Paths** are taken from environment variables (`PIERRE_NWB`, `PIERRE_OUT`, `BIOINFO_DATAPATH`).

## 2. Model and input

- **[A] `InputSpikes_Adapted`** (`datasets/prepare_input.py`): three input channels (constant, GO, NO-GO), exponential decay on the GO and NO-GO
  channels, continuous drive instead of Bernoulli spikes. Replaces `InputSpikes` in `FullModel`.
- **[B] Response-window cue.** A fourth input channel: a pulse at `cue_time` (4.0 s, when the lick port arrives and the response window opens) of
  length `cue_duration` (0.1 s), identical on GO and NO-GO trials, reaching the areas in `cue_areas`. The recordings show a large burst at 4.0 s in both
  areas on every trial, which the model could not produce without an input at that time.
- **[B] Per-channel area routing** in `models/rsnn_nocond_nojawfeedback.py`: sound channels reach only `input_areas` (AC), the cue reaches `cue_areas`
  (ALM and AC). The original sends the input to every unit.
- **[B] `p_exc_in = 1`.** Input channels with index >= `int(p_exc_in * n_rnn_in)` are inhibitory, so with three channels and `p_exc_in` below 1 the
  NO-GO channel inhibits its targets. The original `config_vahid` uses 1; the first adaptation used 0.9.
- **[A, B] `config_vahid`** now describes two areas (ALM, AC), 1000 units (500 per area, 90% excitatory), batch 150, window and onset as in section 1.
  The original uses six areas with 250 units each (200 excitatory, 50 inhibitory).

## 3. Losses

- **[B] Stimulus-stratified losses (`stratify_stim`, on in `config_vahid`).** In the original, neither the neuron-wise PSTH loss (averaged over all trials)
  nor the trial-matching loss (model and data trials are compared by activity only) uses the stimulus. In a GO/NO-GO task that leaves the mapping from input
  to activity unconstrained: GO input producing NO-GO-like activity costs the same as the correct mapping, and a model trained this way learned exactly the
  inverted mapping. Now model trials are matched only to data trials of the same stimulus (one loss term per session and stimulus), and the PSTH loss is
  computed per stimulus with the normalisation taken from all trials (`trial_matching_loss(stims=...)`, `stratified_psth_loss`, `data_stim_tensor` in
  `infopath/losses.py`; `generator_loss(stims=...)` in `infopath/model_loader.py`). Without the option the behaviour is the original one.

## 4. Session stitching

- The sampling code (`infopath/session_stitching.py`) is the original: it fills a model area from the biggest session first and moves to the next session
  only when it runs out. With small sessions this gives a collage over many sessions; with the large sessions of this dataset (417 to 1084 neurons per session
  against 500 model units per area) it uses a single session per area.
- **[B] `spread_sessions` option (on in `config_vahid`)**: the neurons of each (area, excitatory/inhibitory) group are split over all sessions of the area in
  proportion to session size, so that every session constrains the model with its own trials. Neuron identity, area and cell type are preserved. The
  assignment is made once, when the model is built, and saved in the run folder.

## 5. Behaviour

- Original: a linear jaw readout from excitatory units is trained jointly with the network, and a multilayer perceptron applied to the jaw trace classifies
  lick versus no lick.
- Here: **no behaviour in training** (`with_behaviour = False`), so that the recurrent dynamics explain the spikes and not the licks. The behaviour code path of
  the original does not work for this dataset (`with_behaviour=True` is passed to `build_network` as `with_video` and would select no neurons;
  the lick classifier signature was changed by A and no longer matches its call). `infopath/lick_classifier.py` (Poisson count regressor, [A]) is unused.
- **[B] Post-hoc readouts** (`analysis/`): a logistic readout (lick / no lick) and a Poisson multilayer perceptron with two hidden layers of 128 units, as in the
  original paper's classifier, on lick counts from the lick sensor (`analysis/licks.py`). They are trained on the real activity and trials of the neurons
  that are in the model, scored on held-out real trials, and then applied to the model's activity. Controls: shuffled-label null, and the prediction of False Alarm
  versus Correct Rejection within NO-GO trials (beyond the stimulus).

## 6. Training and logging

- **[B]** `--resume <run folder>` (reloads weights and optimizer state, moves the optimizer state to the training device), `--eval_batch_size`,
  `--log_every_n_steps`, `--n_units`, `--batch_size`, `--n_steps` as command line overrides of `infopath/config.py`; GPU peak memory logging.
- **[A]** `infopath/utils/diagnostics.py`: JSONL diagnostics and forward/backward timers in `train.py`.
- **[B]** `infopath/utils/online_logger.py`: optional Comet logging (active only if a key is available) and `infopath/utils/eval_stats.py`: at every evaluation,
  a confusion table of stimulus against assigned trial type, firing rates of model against data (overall, by area, excitatory/inhibitory, session and task
  epoch), and silent fractions.
- **[A, not used by the pipeline above]** `infopath/opto_main.py`, `infopath/opto_own.py`, changes in `infopath/utils/parameter_recovery.py`.
  (A near-duplicate of `functions.py`, `functions_student.py`, was left out: it does not parse and nothing imports it.)

## 7. New analyses and tests (B)

- `analysis/stimulus_check.py` (GO vs NO-GO average activity, model against data), `analysis/lick_decoder_test.py`, `analysis/lick_mlp.py`,
  `analysis/session_repertoire.py` (do the sessions share one repertoire of single-neuron responses; permutation test on cluster composition).
- `tests/test_stratified_matching.py` (the old losses cannot tell a correct from an inverted mapping, the stratified ones can), `tests/test_cue_input.py`
  (cue timing, area routing, input signs), `tests/test_session_spread.py` (collage over all sessions, unique neurons, preserved labels; needs the converted dataset).

## 8. Not included

The recordings, the converted datasets, trained checkpoints of the adapted model and run logs are not part of this repository (see `.gitignore`).
The SLURM scripts in `scripts/slurm/` are templates; partition and QoS names are cluster specific.
