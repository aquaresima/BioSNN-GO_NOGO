# Idea: resample the neurons the model is fitted to (branch `idea/neuron-resampling`)

Status: idea only. No code on this branch beyond this file. `main` has the closest reproduction of the original method (fixed
session collage); this branch records what we would try next and how we would judge it.

## 1. Where we are on `main`

The original method ties every model unit to one specific recorded neuron ("session stitching", Sourmpis et al. 2025, section 4.3):
the units of an area are mapped to neurons recorded in different sessions, which gives a "collage". The assignment is made once,
when the model is built (`infopath/session_stitching.py::build_network`, seeded), saved in the run folder (`sessions.npy`,
`neuron_index.npy`) and never changed.

- The original sampler fills the biggest session first and moves to the next only when it runs out. The paper's datasets had
  small sessions, so this produced a real collage. Here the sessions are large (each has more neurons than the model has units
  per area), so the sampler used ONE session per area and left the other sessions unused.
  The sampling code is the original one (see `CHANGES.md`).
- `main` has `opt.spread_sessions` (on in `config_vahid`): each area's neurons are split over all its sessions in proportion
  to session size, excitatory and inhibitory counts preserved (`tests/test_session_spread.py`).
  The model then contains a random subsample of every session, a small fraction of the recorded neurons.
- Losses that constrain the model (see `CHANGES.md`, section 3): (1) neuron loss, each unit's PSTH against its recorded neuron's,
  separately for GO and NO-GO; (2) trial-matching loss, per session and per stimulus, Sinkhorn distance between the distributions
  of single-trial population signals. The stimulus-stratified versions are `opt.stratify_stim`.

## 2. What is unsatisfying about a fixed collage

- The fixed subset is one random 9 to 15% sample of the recorded neurons; the model reproduces the heterogeneity of that sample.
- Neurons have no identity across sessions, so "unit i is neuron j" is arbitrary; only the distribution of single-neuron
  responses is meaningful.
- The sessions do not share one repertoire of responses (`analysis/session_repertoire.py`: k-means on PSTH shape per area, then a permutation
  test of the cluster composition of each session against random assignment of neurons to sessions): the sessions differ far more than chance
  allows, in cluster composition, in median firing rate and in the fraction of GO/NO-GO selective neurons.
- More sessions do not give more trials per neuron (a neuron lives in one session; its PSTH is estimated from its own session's
  training trials). They give more independent session-level trial sets (one per session instead of one per area).

## 3. The idea

Draw a new set of recorded neurons every K training steps ("epoch"; there are no real epochs in this code, a step is one batch)
and let an assignment step decide which model unit stands for which sampled neuron, instead of fixing the pairing once.
Over training the model then sees the whole population of recorded neurons (Monte Carlo), and what it must reproduce is the
distribution of single-neuron responses, in the spirit of the paper's own sample-and-measure losses.

### 3.1 Neuron-level term (the core of the idea)

1. Precompute for all recorded neurons the GO and NO-GO PSTH (same filter and z-scoring as the neuron loss). Cheap: a few hundred
   time points per neuron and stimulus. No trial tensors needed for this term.
2. Every K steps draw a subset (about 1000, stratified by area and excitatory/inhibitory class, optionally by session).
3. Within each area x E/I class, assign sampled neurons to model units by optimal matching (Hungarian or optimal transport) on the
   distance between the unit's current PSTHs (model, GO and NO-GO) and the neurons' PSTHs.
4. Loss: squared PSTH difference between each unit and its matched neuron, as today.

This asks that the model's units cover the recorded population's range of responses, without tying a unit to one neuron.

### 3.2 Trial-level term (open design problem)

The trial-matching loss needs to know which units belong to which session (it builds each session's population signal per trial).
If assignments change every K steps, those groups change too. Options:

- A. Keep the trial-level term on a fixed collage (as on `main`) and resample only the neuron-level term. Simplest first test.
- B. Re-derive session groups from the assignment at every resample (the units matched to session s define session s's group).
  Needs care: units that were matched to different sessions' neurons now share a population signal.
- C. Replace the per-session grouping by a population-level trial statistic that does not need sessions.

## 4. Risks and open questions

1. Marginals do not give the joint structure: the model can match each unit's PSTH distribution and still get correlations
   between units wrong. The trial-level term is what protects the joint structure, hence the importance of 3.2.
2. Early in training all units are nearly identical, so the optimal matching is nearly arbitrary and resampling adds target noise.
   Resample rarely (every 50 to 100 steps), maybe only after a warm-up with a fixed collage.
3. The sessions differ (section 2). A model fitted to the mixture is a composite animal. This may be what we want, but then
   per-session fit quality will differ and whole-session held-out tests become the meaningful generalisation test.
4. Gradient noise from changing targets; learning-rate schedule interacts with K.
5. Excitatory/inhibitory availability: some sessions have few inhibitory neurons (AC session with 10), so E/I-stratified draws
   must respect availability.

## 5. How we would judge it

All comparisons against the fixed collage on `main`, same architecture, same seeds count, same number of steps:

- Held-out NEURONS (never in any draw) and held-out SESSIONS: do the model's population PSTHs and single-neuron response
  distribution match theirs? The fixed collage cannot offer this test.
- Stimulus dependence: `analysis/stimulus_check.py` (GO vs NO-GO PSTHs, per-neuron effect correlation).
- Behaviour readouts, trained on real data and applied to the model, never in training: `analysis/lick_decoder_test.py`
  (binary lick, shuffle and False Alarm vs Correct Rejection controls) and `analysis/lick_mlp.py` (lick counts, Poisson MLP with
  two hidden layers of 128 units, as in the paper's behaviour classifier).
- Test loss and `t_trial_pearson_ratio` (logged to Comet if enabled); confusion table stimulus vs assigned type.
- Robustness: several seeds (different collages) for the fixed collage, to see how much results depend on one draw.

## 6. Pointers

- Sampler: `infopath/session_stitching.py::build_network` (called once in `infopath/model_loader.py::load_model_and_optimizer`).
- Losses: `infopath/losses.py` (`trial_matching_loss`, `stratified_psth_loss`, `data_stim_tensor`),
  `infopath/model_loader.py::generator_loss`.
- Data layout: `datasets/dataloader.py` (`TrialDataset`, `get_train_trial_type`): trials are per session, padded to the longest.
- Analyses: `analysis/` (run with `scripts/slurm/analysis.sh`).
