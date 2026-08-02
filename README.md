# Time Irreversibility in Cell Motility

A reproducible computational biophysics project for investigating directional asymmetry and time-irreversibility signatures in cell-migration trajectories.

The project uses microscopy-derived trajectories of mesenchymal stem cells and combines Python, statistical analysis, biophysics, automated testing, and reproducible research practices.

> **Current status:** Phase 4 completed. The project is ready for Phase 5: probability-density estimation and displacement-asymmetry analysis.

---

## Scientific Aim

Living cells are active systems that continuously consume energy and operate far from thermodynamic equilibrium.

This project asks whether their recorded motion contains statistical signatures that distinguish forward dynamics from appropriately reversed dynamics.

For a cell trajectory \(\mathbf{r}(t)\), the displacement over a temporal lag \(\tau\) is

\[
\Delta_\tau \mathbf{r}(t)
=
\mathbf{r}(t+\tau)-\mathbf{r}(t).
\]

The project begins by studying asymmetries between observed displacement distributions and their inverted counterparts, while deliberately distinguishing simple directional asymmetry from stronger trajectory-level time irreversibility.

Thermodynamic interpretations such as entropy production are deferred until the required time-reversal controls and statistical validation have been completed.

---

## Dataset

The analysis currently uses sequence `01` of the **Fluo-C2DL-MSC** dataset from the Cell Tracking Challenge.

Cell trajectories were extracted using Fiji / TrackMate.

Current calibration:

- Spatial resolution: `0.3 µm/pixel`
- Temporal resolution: `20 min/frame`
- Sequence length: `48 frames`
- Primary cleaned dataset: `35 cell tracks`

Processed biological data are kept locally and are not normally committed to the public repository.

---

## Current Analysis Pipeline

The project has completed the following stages:

- **Phase 1:** reproducible Python/Conda project setup
- **Phase 2:** TrackMate trajectory cleaning and standardization
- **Phase 3:** multi-lag displacement computation
- **Phase 4:** temporal-gap audit, exact-lag validation, common-motion analysis, and construction of a co-moving representation

Displacements are currently calculated at:

| Lag | Physical time | Exact steps |
|---:|---:|---:|
| 1 frame | 20 min | 433 |
| 2 frames | 40 min | 398 |
| 4 frames | 80 min | 343 |

---

## Important Phase 3 Correction

An audit identified that the original displacement implementation used row-based shifting, which can assign the wrong physical lag when intermediate frames are missing.

The pipeline was therefore redesigned to match observations using exact frame numbers:

```python
frame_end = frame_start + tau_frames
```

Only observations belonging to the same cell at the exact requested endpoint frame are retained.

The corrected implementation is contained in:

```text
src/steps.py
```

and is protected by automated regression tests.

---

## Common Motion and Co-Moving Coordinates

Phase 4 detected a population-level directional component, particularly along the x-axis.

Because trajectory data alone cannot determine whether this reflects microscope drift, collective biological migration, or a mixture of both, the project does **not** automatically label it as technical drift.

Instead, two parallel representations are retained:

```text
Raw trajectories
Co-moving trajectories
```

The co-moving coordinates remove the frame-wise median population translation and are used as a sensitivity analysis.

Future irreversibility results will therefore be compared between raw and co-moving data.

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

The validated development environment uses:

```text
Python 3.11.13
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

The tests include regression checks for exact physical frame lags, missing frames, duplicate cell-frame observations, multi-lag displacement calculations, and trajectory integrity.

---

## Detailed Documentation

The complete scientific and educational record of the project is maintained in:

**[Cell Motility Irreversibility Master Handbook](instructions/Cell_Motility_Irreversibility_Master_Handbook.pdf)**

The handbook contains the detailed material intentionally omitted from this README, including:

- full project history;
- TrackMate workflow and data cleaning;
- Python code and programming concepts;
- mathematical derivations;
- biophysical interpretation;
- the exact-lag bug and its diagnosis;
- Phase 4 common-motion analysis;
- raw versus co-moving rationale;
- reproducibility and environment setup;
- automated testing;
- Git/GitHub workflow;
- scientific decisions and rejected alternatives;
- and the roadmap for subsequent phases.

---

## Next Phase

**Phase 5** will begin the main statistical asymmetry analysis, including:

- displacement-density estimation;
- comparison of \(p(\Delta)\) and \(p(-\Delta)\);
- raw versus co-moving results;
- uncertainty estimation and bootstrap confidence intervals;
- and appropriate symmetry/time-reversal controls.

---

## Repository

https://github.com/AbolDpirate/cell-irreversibility