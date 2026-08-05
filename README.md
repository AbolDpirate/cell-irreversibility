# Time Irreversibility in Cell Motility

A reproducible computational biophysics project for exploring directional asymmetry and time-irreversibility signatures in cell-migration trajectories.

The project uses microscopy-derived trajectories of mesenchymal stem cells and is developed both as:

- an exploratory scientific analysis; and
- a structured learning project in Python, scientific computing, statistics, biophysics, reproducible research, testing, and Git/GitHub.

> **Current status:** Phase 5 completed.  
> **Next:** Phase 6 — ordered multi-step sequence analysis and trajectory-level time reversal.

---

## Scientific Aim

Living cells are active systems that continuously consume energy and operate far from thermodynamic equilibrium.

This project asks whether experimentally measured cell trajectories contain statistical signatures that distinguish forward dynamics from appropriately reversed dynamics.

For a cell trajectory, the displacement over lag `tau` is

$$
\Delta_\tau \mathbf{r}(t)
=
\mathbf{r}(t+\tau)-\mathbf{r}(t).
$$

The project first studies spatial inversion asymmetry by comparing the probability of an observed displacement with that of its inverted counterpart:

$$
p(\Delta \mathbf{r})
\quad \text{vs.} \quad
p(-\Delta \mathbf{r}).
$$

This is treated as an exploratory displacement-asymmetry analysis, not yet as definitive trajectory-level time irreversibility or thermodynamic entropy production.

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

## Current Analysis Pipeline

Completed stages:

- **Phase 1:** reproducible Python/Conda project setup
- **Phase 2:** TrackMate trajectory cleaning and standardization
- **Phase 3:** exact-lag multi-scale displacement computation
- **Phase 4:** temporal-gap audit, common-motion analysis, and co-moving coordinates
- **Phase 5:** displacement distributions, density estimation, and spatial inversion-asymmetry analysis

Current exact displacement counts are:

| Lag | Physical time | Exact steps |
|---:|---:|---:|
| 1 frame | 20 min | 433 |
| 2 frames | 40 min | 398 |
| 4 frames | 80 min | 343 |

---

## Exact-Lag Correction

A Phase 4 audit identified that the original displacement implementation used row-based shifting, which can assign incorrect physical lags when trajectory frames are missing.

The authoritative implementation now matches endpoints using exact frame numbers:

```python
frame_end = frame_start + tau_frames
```

and joins observations belonging to the same cell at the required endpoint frame.

The corrected implementation is maintained in:

```text
src/steps.py
```

and protected by automated regression tests.

---

## Raw and Co-Moving Trajectories

A population-level common-motion component was detected, particularly along the x direction.

Because trajectory data alone cannot establish whether this represents microscope drift, collective biological migration, or a combination of both, the project does not treat it as definitively technical drift.

Two representations are therefore retained:

```text
Raw trajectories
Co-moving trajectories
```

The co-moving representation subtracts the frame-wise common population translation and is treated as a sensitivity analysis rather than an automatically superior dataset.

---

## Phase 5 — Displacement Asymmetry

Phase 5 characterized the empirical displacement distributions using:

- normalized histograms;
- Freedman-Diaconis data-driven binning;
- one-dimensional kernel-density estimation;
- two-dimensional KDE of `p(dx, dy)`;
- comparison of `p(Delta)` with `p(-Delta)`;
- KDE-bandwidth sensitivity;
- cell-level bootstrap uncertainty;
- an inversion-symmetrized null control;
- and a sensitivity analysis for tracks with constant raw `y` coordinates.

The exploratory inversion-asymmetry quantity was based on

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

The raw representation showed greater inversion asymmetry than the co-moving representation at all three lags.

This qualitative ordering remained unchanged when the KDE bandwidth was varied between `0.5×`, `1×`, and `2×` the baseline value, although the absolute numerical magnitude of the metric was bandwidth-dependent.

A simple inversion-symmetrized null comparison found:

| `tau` | Raw above null 95th percentile | Co-moving above null 95th percentile |
|---:|:---:|:---:|
| 1 | Yes | No |
| 2 | Yes | Yes |
| 4 | Yes | Yes |

These results are interpreted as **exploratory evidence of single-displacement spatial inversion asymmetry**.

They do not establish full temporal irreversibility or entropy production.

---

## Data-Quality and Interpretation Note

Exploratory diagnostics identified several tracks with constant raw `y` coordinates and some large overlapping multi-frame displacements.

These observations were retained in the primary analysis because the project is intended as an exploratory and educational scientific workflow rather than a publication-grade tracking-validation study.

A sensitivity analysis excluding the constant-`y` tracks reduced the estimated asymmetry but did not remove the overall qualitative pattern.

Known data limitations are therefore documented explicitly and incorporated into interpretation rather than treated as grounds for extensive forensic re-analysis.

---

## Repository Structure

```text
cell-irreversibility/
│
├── data/                 # local/raw and processed data
├── envs/                 # environment snapshots
├── figures/              # generated figures
├── instructions/         # detailed project handbook
├── notebooks/            # phase-specific analyses
├── src/                  # reusable scientific code
├── tests/                # automated tests
├── env.yml               # Conda environment
└── README.md
```

Important notebooks currently include:

```text
02_clean_spots_MSC01.ipynb
03_compute_steps_MSC01.ipynb
04_drift_validation_MSC01.ipynb
05_displacement_density_MSC01.ipynb
```

Core source modules:

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

Run the automated tests from the repository root:

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

The full scientific and educational project record is maintained in:

**[Cell Motility Irreversibility Master Handbook](instructions/Cell_Motility_Irreversibility_Master_Handbook.pdf)**

The handbook contains the details intentionally omitted from this README, including:

- project history and scientific decisions;
- TrackMate workflow and data cleaning;
- Python and scientific-computing concepts;
- mathematical and statistical derivations;
- exact-lag debugging and validation;
- common-motion and co-moving analysis;
- density-estimation methodology;
- bootstrap and null-control logic;
- biophysical interpretation;
- reproducibility and Git/GitHub workflow;
- limitations and sensitivity analyses;
- and the phase-by-phase project roadmap.

---

## Next Phase

Phase 6 will move beyond single-displacement spatial inversion and examine **ordered multi-step displacement sequences**.

For example, a forward sequence

$$
(\Delta_1,\Delta_2,\Delta_3)
$$

has the time-reversed counterpart

$$
(-\Delta_3,-\Delta_2,-\Delta_1).
$$

This introduces temporal ordering explicitly and therefore moves the project closer to a genuine analysis of the statistical arrow of time.

---

## Repository

https://github.com/AbolDpirate/cell-irreversibility

---

## Status

**Phase 5 completed. Phase 6 — sequence-level time reversal — is next.**