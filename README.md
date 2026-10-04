# BioSNN-GO_NOGO

A biologically informed spiking recurrent network (bioRNN) fitted to electrophysiological recordings from a **delayed auditory GO/NO-GO task**
in the anterior lateral motor cortex (ALM) and the auditory cortex (AC) of the mouse. The network is a multi-area LIF model with Dale's law,
whose units are tied to recorded neurons; it is trained to reproduce the recorded activity at the level of single-neuron PSTHs and of the
distribution of single-trial population activity. The goal is to use the fitted model to study the dynamics that hold stimulus identity during
the memory delay.

This repository started as a fork of [Sourmpis/BiologicallyInformed](https://github.com/Sourmpis/BiologicallyInformed), the code of

> C. Sourmpis, C. Petersen, W. Gerstner & G. Bellec,
> *Biologically informed cortical models predict optogenetic perturbations*, bioRxiv 2024,
> [doi:10.1101/2024.09.27.615361](https://www.biorxiv.org/content/10.1101/2024.09.27.615361)

and has been adapted to a different dataset and task. What changed with respect to the original project is listed in [CHANGES.md](CHANGES.md).

## Scope

- **Task.** Each trial has a 2 s sound (GO or NO-GO), a 2 s silent delay, and a response window that opens at 4 s when the lick port arrives.
  The mouse licks for GO trials (Hit) and withholds for NO-GO trials (Correct Rejection); errors are Miss and False Alarm.
- **Data.** Neuropixels recordings in NWB format, one area per session (several ALM and several AC sessions). **The recordings are not part of this
  repository and are not distributed with it.** `pierre/process.py` documents the format it expects (spike times, trial table, lick sensor, sound copy).
- **Model.** 1000 units (500 per area, 90% excitatory), four input channels (constant, GO, NO-GO, and a response-window cue at 4 s), recurrent
  spiking dynamics with surrogate gradients, trained with a neuron-wise PSTH loss and a trial-matching (optimal transport) loss, both computed
  per stimulus.
- **Session collage.** As in the original method, every model neuron is mapped to one recorded neuron; neurons of an area are spread over all
  sessions of that area so that every session constrains the model with its own trials.
- **Behaviour.** Lick behaviour is deliberately **not** used in training. It is read out afterwards from the spiking activity with readouts trained on
  the real data and applied to the model (`analysis/`).
- **Status.** Work in progress. The pipeline runs end to end; fits of the collage model are being evaluated and no results are claimed here.

An idea that is not implemented on `main` is described on the branch `idea/neuron-resampling` (resample the recorded neurons the model is fitted to
during training).

## Installation

Same environment as the original project: `bash setup.sh` (conda environment `bioinfo`, Python 3.10) or `pip install -e .`.
Additional packages used here: `scikit-learn`, `scipy`, `pandas`, `h5py`, and optionally `comet_ml` for online logging
(active only if a Comet API key is available in the environment or in `~/.comet.config`).
Training needs a GPU with at least 48 GB of memory at the default size (measured peak 38.8 GB).

## Pipeline

1. Convert the NWB sessions: `PIERRE_NWB=<folder with .nwb> PIERRE_OUT=<output> python pierre/process.py`.
2. Point the code to the converted dataset: `export BIOINFO_DATAPATH=<output>`.
3. Create a configuration and train:
   ```bash
   python infopath/config.py --config run1 --n_units 1000 --batch_size 150 --n_steps 20000 --log_every_n_steps 100 --eval_batch_size 200
   python infopath/train.py --config run1            # add --resume <log_dir/.../run folder> to continue a run
   ```
   `scripts/slurm/` has SLURM templates (partition and QoS names are cluster specific and must be adapted).
4. Analyse a checkpoint (about one minute each): `python -m analysis.stimulus_check --run <run folder> --out <out>`,
   `python -m analysis.lick_decoder_test ...`, `python -m analysis.lick_mlp ...`; `python -m analysis.session_repertoire --data $BIOINFO_DATAPATH --out <out>`
   compares the sessions' response repertoires.

## Layout

| Path | Contents |
|---|---|
| `infopath/` | training loop, losses, model loader, configuration, logging |
| `models/` | the recurrent spiking network |
| `datasets/` | data loader and input encoding (`prepare_input.py`) |
| `pierre/` | NWB converter for the GO/NO-GO dataset |
| `analysis/` | post-hoc analyses of checkpoints and of the data |
| `tests/` | unit tests of the stratified losses, the cue channel and the session collage |
| `scripts/slurm/` | SLURM templates |
| `Figures/`, `AllModels/`, `configs/pseudo_data/` | from the original project (paper figures, pretrained models, synthetic-data configurations) |

## Acknowledgements and license

The original code and method are by C. Sourmpis, C. Petersen, W. Gerstner and G. Bellec. The first adaptation to the GO/NO-GO data (converter,
sound encoding, configuration for ALM and AC) was done by Daniel Carrillos during an internship at the Institut Pasteur / Institut de l'Audition.
MIT license, as the original (see `LICENSE`).
