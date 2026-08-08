# Time Irreversibility in Cell Motility

A reproducible computational biophysics project for exploring directional asymmetry and time-reversal signatures in cell-migration trajectories.

The project is developed both as:

- an exploratory scientific analysis; and
- a structured learning project in Python, scientific computing, statistics, biophysics, reproducible research, testing, and Git/GitHub.

> **Current status:** Phase 6 completed.  
> **Next:** Phase 7 — extending temporal irreversibility analysis beyond the current two-step feature representation.

---

## Scientific Aim

Living cells are active systems that continuously consume energy and operate away from thermodynamic equilibrium.

This project asks whether experimentally measured cell trajectories contain statistical signatures that distinguish forward dynamics from appropriately reversed dynamics.

For a cell trajectory, the displacement over lag `tau` is

$$
\Delta_\tau \mathbf{r}(t)
=
\mathbf{r}(t+\tau)-\mathbf{r}(t).
$$

The project progressively moves from single-displacement spatial asymmetry toward explicitly time-ordered trajectory analysis.

The current results are treated as exploratory and hypothesis-generating rather than as definitive measurements of biological entropy production or a thermodynamic arrow of time.

---

## Dataset

The current analysis uses sequence `01` of the **Fluo-C2DL-MSC** dataset from the Cell Tracking Challenge.

Cell trajectories were extracted using Fiji / TrackMate.

Current calibration:

- Spatial resolution: `0.3 µm/pixel`
- Temporal resolution: `20 min/frame`
- Sequence length: `48 frames`
- Primary cleaned dataset: `35 cell tracks`

Raw and processed biological datasets are kept locally and are not normally committed to the public repository.

---

## Project Progress

Completed stages:

- **Phase 1:** reproducible Python/Conda project setup
- **Phase 2:** TrackMate trajectory cleaning and standardization
- **Phase 3:** exact-lag multi-scale displacement computation
- **Phase 4:** temporal-gap audit, common-motion analysis, and co-moving coordinates
- **Phase 5:** displacement distributions and single-step spatial inversion asymmetry
- **Phase 6:** ordered two-step sequence construction and exploratory time-reversal analysis

---

## Exact-Lag Displacements

A previous audit identified that row-based shifting can produce incorrect physical lags when trajectory frames are missing.

The authoritative implementation now matches trajectory endpoints using exact frame numbers.

Current exact displacement counts are:

| Lag | Physical time | Exact steps |
|---:|---:|---:|
| 1 frame | 20 min | 433 |
| 2 frames | 40 min | 398 |
| 4 frames | 80 min | 343 |

The corrected implementation is maintained in:

```text
src/steps.py
```

and protected by automated regression tests.

---

## Raw and Co-Moving Representations

A population-level common-motion component was detected, particularly along the x direction.

Trajectory data alone cannot establish whether this common motion represents microscope drift, collective biological migration, or a combination of both.

The project therefore retains two parallel representations:

```text
Raw trajectories
Co-moving trajectories
```

The co-moving representation subtracts the frame-wise common population translation and is treated as a sensitivity representation rather than as an automatically superior reconstruction of the true cell motion.

---

## Phase 5 — Single-Displacement Spatial Inversion

Phase 5 analyzed the empirical displacement distribution

$$
p(\Delta \mathbf{r})
$$

and compared it with

$$
p(-\Delta \mathbf{r}).
$$

The analysis used:

- empirical histograms;
- Freedman-Diaconis binning;
- one-dimensional KDE;
- two-dimensional KDE;
- raw versus co-moving comparison;
- bandwidth sensitivity;
- cell-level bootstrap uncertainty;
- an inversion-symmetrized null control;
- and sensitivity analysis for constant-y tracks.

The exploratory inversion-asymmetry quantity was

$$
I_\tau
=
D_{\mathrm{KL}}
\left[
p(\Delta)
\parallel
p(-\Delta)
\right].
$$

Baseline estimates were:

| `tau` | Raw | Co-moving |
|---:|---:|---:|
| 1 | 0.682 | 0.483 |
| 2 | 0.568 | 0.438 |
| 4 | 0.793 | 0.542 |

A simple inversion-symmetrized null comparison found:

| `tau` | Raw above null 95th percentile | Co-moving above null 95th percentile |
|---:|:---:|:---:|
| 1 | Yes | No |
| 2 | Yes | Yes |
| 4 | Yes | Yes |

These results were interpreted as exploratory evidence of **single-displacement spatial inversion asymmetry**.

However, spatial inversion of a single displacement does not by itself test temporal ordering.

---

## Phase 6 — Ordered Sequence Time Reversal

Phase 6 introduced explicit temporal ordering using exact two-step sequences constructed from three consecutive observations:

$$
\mathbf{r}_t,
\quad
\mathbf{r}_{t+1},
\quad
\mathbf{r}_{t+2}.
$$

A forward sequence was defined as

$$
S =
(\Delta \mathbf{r}_1,\Delta \mathbf{r}_2),
$$

with the correct time-reversed sequence

$$
R(S)
=
(-\Delta \mathbf{r}_2,-\Delta \mathbf{r}_1).
$$

The implementation was computationally validated to satisfy

$$
R(R(S))=S.
$$

A total of:

```text
392 exact two-step sequences
32 contributing cells
```

were constructed for both the raw and co-moving trajectories.

---

## Phase 6 Feature Representation

Direct density estimation in the full four-dimensional sequence space

$$
(\Delta x_1,\Delta y_1,\Delta x_2,\Delta y_2)
$$

would be relatively data-hungry for the available sample size.

Phase 6 therefore used two interpretable time-odd sequence features:

### Change in step magnitude

$$
\Delta m = m_2-m_1
$$

and a signed turning representation based on

$$
\sin(\theta).
$$

Under time reversal,

$$
(\Delta m,\sin\theta)
\longrightarrow
(-\Delta m,-\sin\theta).
$$

After requiring valid turning information in both raw and co-moving representations, the primary paired analysis used:

```text
365 sequences
31 cells
```

---

## Phase 6 Results

The baseline KDE-based sequence asymmetry estimates were:

| Representation | Sequence asymmetry (nats) |
|---|---:|
| Raw | 0.107 |
| Co-moving | 0.189 |

The absolute values were strongly bandwidth-dependent.

The co-moving estimate was larger than the raw estimate across the tested bandwidths, but cell-level bootstrap uncertainty for the difference included zero:

$$
I_{\mathrm{comoving}}
-
I_{\mathrm{raw}}
\approx
[-0.033,\ 0.184]
$$

for the exploratory 95% bootstrap interval.

Most importantly, the observed values were compared with a time-reversal-symmetrized null distribution.

| Representation | Observed | Null mean | Null 95th percentile | Above null 95th? |
|---|---:|---:|---:|:---:|
| Raw | 0.107 | 0.191 | 0.264 | No |
| Co-moving | 0.189 | 0.185 | 0.262 | No |

Neither observed estimate exceeded the corresponding null 95th percentile.

Therefore, within the current two-step feature representation and MSC01 dataset, Phase 6 did **not detect sequence-level time-reversal asymmetry beyond the finite-sample/KDE background**.

This does not demonstrate that the underlying cell dynamics are time-reversible.

It only means that the present two-step, two-feature analysis did not provide detectable evidence of irreversibility beyond the chosen null model.

---

## Scientific Interpretation

The project currently supports two distinct observations:

1. single-displacement spatial inversion asymmetry can be detected in parts of the Phase 5 analysis;
2. the more temporally explicit Phase 6 sequence analysis did not exceed its finite-sample time-reversal-symmetrized null background.

This distinction is scientifically important.

A positive asymmetry estimator alone is not sufficient evidence for an arrow of time.

Finite sample size, density estimation, representation choice, temporal resolution, sequence length, and dimensionality all affect what can be detected.

The current results therefore remain exploratory and should not be interpreted as measurements of entropy production or definitive non-equilibrium thermodynamic quantities.

---

## Repository Structure

```text
cell-irreversibility/
│
├── data/                 # local/raw and processed data
├── envs/                 # environment snapshots
├── figures/              # generated figures
├── instructions/         # detailed project handbooks
├── notebooks/            # phase-specific analyses
├── src/                  # reusable scientific code
├── tests/                # automated tests
├── env.yml               # Conda environment
└── README.md
```

Current notebooks include:

```text
02_clean_spots_MSC01.ipynb
03_compute_steps_MSC01.ipynb
04_drift_validation_MSC01.ipynb
05_displacement_density_MSC01.ipynb
06_phase6_sequence_irreversibility_MSC01.ipynb
```

Core source modules include:

```text
src/io.py
src/steps.py
src/density.py
src/metrics.py
src/plots.py
```

---

## Environment

The validated development environment uses Python 3.11 and includes:

```text
NumPy
pandas
Matplotlib
scikit-learn
Jupyter / ipykernel
pytest
```

Create the Conda environment with:

```bash
conda env create -f env.yml
```

Activate it with:

```bash
conda activate cell-irreversibility
```

---

## Tests

Run the automated test suite from the repository root:

```bash
python -m pytest -q
```

Current validated status:

```text
9 passed
```

The test suite includes regression checks for exact physical frame lags, missing frames, duplicate cell-frame observations, multi-lag displacement calculations, and trajectory integrity.

---

## Detailed Documentation

Detailed project documentation is maintained in the `instructions/` directory.

### Phases 0–4

**[Cell Motility Irreversibility Master Handbook](instructions/Cell_Motility_Irreversibility_Master_Handbook.pdf)**

### Phase 5

**[Standalone Phase 5 Handbook](instructions/Cell_Motility_Irreversibility_Phase5_Standalone_Handbook.pdf)**

The handbooks contain substantially more detail than this README, including:

- project history;
- code-level workflow;
- mathematical derivations;
- Python concepts;
- statistical reasoning;
- biophysical interpretation;
- debugging and validation;
- scientific decision points;
- limitations;
- sensitivity analyses;
- reproducibility practices;
- and continuation notes for later project phases.

---

## Next Phase

Phase 7 will investigate whether temporal information becomes more detectable when the analysis moves beyond the current reduced two-step representation.

Possible directions include:

- richer representations of the original sequence coordinates;
- longer ordered sequences;
- alternative low-dimensional time-odd observables;
- and methods that avoid direct high-dimensional KDE where appropriate.

The exact Phase 7 design will be chosen based on interpretability, sample size, and the educational scope of the project rather than by forcing a positive irreversibility result.

---

## Repository

https://github.com/AbolDpirate/cell-irreversibility

---

## Status

**Phase 6 completed. Phase 7 is next.**