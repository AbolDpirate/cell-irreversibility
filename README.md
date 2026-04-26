# Time Irreversibility in Cell Motility

This project investigates statistical time irreversibility in living cell motion using cell-tracking data. The main idea is to convert cell trajectories into displacement steps and compare the probability of forward steps, `p(Delta)`, with their mirrored counterparts, `p(-Delta)`. A difference between these distributions can indicate a statistical arrow of time and non-equilibrium behavior in cellular dynamics.

## Project Motivation

Living cells consume energy and operate far from thermodynamic equilibrium. This project uses real microscopy-based cell trajectories to explore whether this non-equilibrium behavior can be detected from motion alone.

The core quantity of interest is an irreversibility index:

```text
I_tau = < log( p(Delta_tau) / p(-Delta_tau) ) >
```

where `Delta_tau` is the displacement of a cell over a time lag `tau`.

## Dataset

The project uses the **Fluo-C2DL-MSC** dataset from the **Cell Tracking Challenge (CTC)**, specifically sequence `01`.

The dataset contains fluorescence microscopy images of mesenchymal stem cells. Cell positions were extracted from the raw image sequence using **Fiji / TrackMate**.

## Current Workflow

### Phase 1 — Environment and Project Setup

A reproducible Python project structure was created with:

- `src/` for reusable Python modules
- `notebooks/` for step-by-step analysis
- `data/` for raw and processed data
- `figures/` for generated plots
- `tests/` for basic import tests
- `env.yml` for the Conda environment

The main environment can be created with:

```bash
conda env create -f env.yml
conda activate cell-irreversibility
```

### Phase 2 — Data Extraction and Cleaning

The raw CTC image sequence was processed in **Fiji / TrackMate**.

TrackMate settings included:

- Detector: LoG detector
- Estimated object diameter: 28 µm
- Quality threshold: 2
- Median filter: enabled
- Sub-pixel localization: enabled
- Tracker: Simple LAP tracker
- Linking max distance: 8 µm
- Gap-closing max distance: 15 µm
- Gap-closing max frame gap: 2

The raw TrackMate Spots CSV was cleaned with pandas into standardized trajectory files:

```text
data/MSC01_tracks_clean.csv
data/MSC01_tracks_clean_min10.csv
```

The cleaned trajectory format is:

```text
cell_id, frame, t_min, x_um, y_um
```

### Phase 3 — Displacement Step Computation

Cell trajectories were converted into displacement steps for:

```text
tau = 1, 2, 4 frames
```

With a 20-minute frame interval, these correspond to:

```text
20, 40, and 80 minutes
```

The output step files include:

```text
data/MSC01_steps_tau124.csv
data/MSC01_steps_tau1.csv
data/MSC01_steps_tau2.csv
data/MSC01_steps_tau4.csv
data/MSC01_steps_summary_tau124.csv
```

Initial diagnostics showed likely image/stage drift, based on mean displacement scaling with lag time.

### Current Status

The project is currently at the beginning of **Phase 4**.

Completed:

- Project structure and Conda environment
- TrackMate extraction from CTC sequence 01
- Cleaning of raw TrackMate CSV
- Construction of displacement steps for `tau = 1, 2, and 4`
- Initial drift diagnostics

Next steps:

- Drift correction
- Estimation of `p(Delta)` and `p(-Delta)`
- Computation of `I_tau`
- Sign-shuffle and time-reversal controls
- Robustness checks for histogram binning and smoothing parameters

## Repository Structure

```text
cell-irreversibility/
  data/
  figures/
  notebooks/
  src/
    io.py
    steps.py
    density.py
    metrics.py
    plots.py
  tests/
  env.yml
  README.md
```

## Main Python Dependencies

- Python 3.11
- numpy
- pandas
- matplotlib
- scikit-learn
- jupyterlab
- pytest
- ipykernel

## Reproducibility

To recreate the environment:

```bash
conda env create -f env.yml
conda activate cell-irreversibility
```

To run basic tests:

```bash
pytest -q
```

## Notes

This is an active learning and research project. The current focus is on building a transparent, reproducible pipeline for extracting cell motion statistics and connecting them to concepts from non-equilibrium biophysics.
