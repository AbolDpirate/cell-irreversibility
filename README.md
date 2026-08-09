# Time Irreversibility in Cell Motility

A reproducible computational biophysics project for exploring directional asymmetry and time-reversal signatures in cell-migration trajectories.

The project is developed both as:

- an exploratory scientific analysis; and
- a structured learning project in Python, scientific computing, statistics, biophysics, machine learning, reproducible research, testing, and Git/GitHub.

> **Current status:** Phase 7 completed.  
> **Next:** Phase 8 — extending temporal analysis beyond the present two-step sequence framework.

---

## Scientific Aim

Living cells are active systems that continuously consume energy and operate away from thermodynamic equilibrium.

This project asks whether experimentally measured cell trajectories contain statistical information that distinguishes forward dynamics from appropriately time-reversed dynamics.

For a cell trajectory, a displacement over lag `tau` is

$$
\Delta_\tau \mathbf{r}(t)
=
\mathbf{r}(t+\tau)-\mathbf{r}(t).
$$

The project progressively moves from single-displacement spatial asymmetry toward explicitly ordered trajectory analysis.

The current results are exploratory and should not be interpreted as direct measurements of entropy production or a thermodynamic arrow of time.

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
- **Phase 6:** ordered two-step sequence construction and reduced-feature time-reversal analysis
- **Phase 7:** full-sequence forward/reverse classification with grouped cross-validation and time-reversal null controls

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

Trajectory data alone cannot establish whether this component represents:

- microscope or stage drift;
- collective biological migration;
- or a combination of both.

The project therefore retains two parallel representations:

```text
Raw trajectories
Co-moving trajectories
```

The co-moving representation subtracts frame-wise common population translation.

It is treated as a sensitivity representation rather than as an automatically superior reconstruction of the true biological motion.

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

However, spatial inversion of an individual displacement does not by itself test temporal ordering.

---

## Phase 6 — Ordered Two-Step Time Reversal

Phase 6 introduced explicit temporal ordering using three consecutive observations:

$$
\mathbf{r}_t,
\quad
\mathbf{r}_{t+1},
\quad
\mathbf{r}_{t+2}.
$$

The ordered two-step sequence was defined as

$$
S=
(\Delta\mathbf{r}_1,\Delta\mathbf{r}_2),
$$

with time reversal

$$
R(S)
=
(-\Delta\mathbf{r}_2,-\Delta\mathbf{r}_1).
$$

The involution property

$$
R(R(S))=S
$$

was computationally verified.

The exact sequence dataset contained:

```text
392 two-step sequences
32 cells
```

Phase 6 used two interpretable time-odd features:

$$
\Delta m=m_2-m_1
$$

and

$$
\sin(\theta).
$$

After requiring valid turning-angle information in both representations, the paired feature analysis contained:

```text
365 sequences
31 cells
```

Baseline KDE asymmetry estimates were:

| Representation | Sequence asymmetry |
|---|---:|
| Raw | 0.107 |
| Co-moving | 0.189 |

However, neither observed value exceeded the corresponding time-reversal-symmetrized null 95th percentile.

Therefore, Phase 6 did **not detect sequence-level time-reversal asymmetry beyond the finite-sample/KDE background in the selected reduced representation**.

This result motivated Phase 7.

---

## Phase 7 — Full-Sequence Classification

Phase 7 asked whether Phase 6 might have missed temporal information because the original four-dimensional sequence

$$
(\Delta x_1,\Delta y_1,\Delta x_2,\Delta y_2)
$$

had been reduced to only two hand-designed features.

The full two-step sequence was therefore retained.

For every observed forward sequence,

$$
S,
$$

the correctly time-reversed sequence

$$
R(S)
$$

was generated.

This produced a balanced classification dataset containing:

```text
392 forward sequences
392 reversed sequences
784 total classifier examples
32 biological cells
```

Forward sequences were assigned label `1` and reversed sequences label `0`.

---

## Cell-Grouped Cross-Validation

Sequences from the same cell are statistically related.

A naive random train/test split could therefore leak information from individual biological cells between training and testing.

Phase 7 used:

```text
5-fold GroupKFold
```

with `cell_id` as the grouping variable.

All forward and reversed examples from a given cell remained in the same fold.

Thus, when a cell was used for testing, no sequence from that cell appeared in the training data.

---

## Linear Classification

The primary linear classifier was:

```text
StandardScaler
      ↓
LogisticRegression
```

Scaling was performed inside the scikit-learn pipeline so that preprocessing was fitted using training cells only.

Mean grouped-cross-validation performance was:

| Representation | Mean ROC AUC | Mean balanced accuracy |
|---|---:|---:|
| Raw | 0.622 | 0.614 |
| Co-moving | 0.504 | 0.520 |

The raw classifier initially appeared to contain detectable time-direction information.

A paired time-reversal-symmetrized null analysis with 200 replicates gave:

| Metric | Observed | Null mean | Null 95th percentile | Empirical upper-tail `p` |
|---|---:|---:|---:|---:|
| Raw ROC AUC | 0.622 | 0.500 | 0.556 | 0.005 |
| Co-moving ROC AUC | 0.504 | 0.502 | 0.559 | 0.512 |

The raw classifier therefore exceeded the paired null background, whereas the co-moving classifier did not.

However, this result required further interpretation.

---

## Reversal-Odd / Reversal-Even Decomposition

For a sequence

$$
S=(\mathbf d_1,\mathbf d_2),
$$

Phase 7 defined the reversal-odd coordinate

$$
\mathbf o
=
\mathbf d_1+\mathbf d_2
$$

and the reversal-even coordinate

$$
\mathbf e
=
\mathbf d_1-\mathbf d_2.
$$

Under time reversal,

$$
\mathbf o\rightarrow-\mathbf o,
$$

whereas

$$
\mathbf e\rightarrow\mathbf e.
$$

Importantly,

$$
\mathbf d_1+\mathbf d_2
=
\mathbf r_{t+2}-\mathbf r_t.
$$

Thus, the odd coordinate is simply the **net displacement across the two-step interval**.

Linear classification produced:

| Representation | Feature space | Mean ROC AUC |
|---|---|---:|
| Raw | Full 4D | 0.622 |
| Raw | Odd net displacement | 0.622 |
| Raw | Even step difference | 0.500 |
| Co-moving | Full 4D | 0.504 |
| Co-moving | Odd net displacement | 0.504 |
| Co-moving | Even step difference | 0.500 |

The full raw model and odd-only model were therefore essentially identical.

This demonstrates that the significant raw linear classification signal was almost entirely explained by **net directional displacement**, rather than by richer two-step sequence ordering.

The disappearance of this signal in the co-moving representation further indicates that common population motion contributes strongly to the raw classification result.

---

## Nonlinear Sensitivity Analysis

One pre-specified nonlinear model was tested:

```text
StandardScaler
      ↓
RBF-SVM
```

No broad model search or aggressive hyperparameter optimization was performed.

Observed performance was:

| Representation | Feature space | Mean ROC AUC |
|---|---|---:|
| Raw | Full 4D | 0.557 |
| Raw | Odd net displacement | 0.565 |
| Co-moving | Full 4D | 0.348 |
| Co-moving | Odd net displacement | 0.371 |

The corresponding paired-null analysis produced:

| Metric | Observed | Empirical upper-tail `p` |
|---|---:|---:|
| Raw full 4D | 0.557 | 0.075 |
| Raw odd-only | 0.565 | 0.055 |
| Raw full − odd | -0.008 | 0.602 |
| Co-moving full 4D | 0.348 | 1.000 |
| Co-moving odd-only | 0.371 | 1.000 |
| Co-moving full − odd | -0.023 | 0.811 |

The raw odd-only result was borderline relative to the estimated null 95th percentile but did not provide a robust empirical upper-tail result.

Most importantly, the full four-dimensional representation did **not** outperform the odd net-displacement representation in either raw or co-moving trajectories.

Below-chance co-moving RBF performance was retained as observed and was not post-hoc inverted.

---

## Phase 7 Interpretation

Phase 7 found that:

1. a linear classifier can distinguish raw forward and reversed sequences above the paired null background;
2. almost all of that linear discrimination is reproduced by the reversal-odd net-displacement coordinate;
3. the linear signal largely disappears after common-motion subtraction;
4. reversal-even features alone contain no classification information, as expected;
5. a nonlinear RBF classifier did not reveal convincing additional information in the full four-dimensional sequence;
6. full-sequence RBF classification did not outperform net displacement alone.

Therefore, Phase 7 does **not provide convincing evidence for intrinsic two-step sequence-order irreversibility beyond common directional motion**.

The raw directional signal remains scientifically relevant, but trajectory data alone cannot determine whether the common-motion component reflects technical drift, collective migration, or both.

The results also do not demonstrate that the underlying cell dynamics are time-reversible.

They only show that the two-step analyses performed so far have not detected a robust sequence-order signal after accounting for common population motion and finite-sample null behavior.

No entropy-production or thermodynamic-arrow interpretation is justified from these results alone.

---

## Machine Learning Scope

Machine learning is used in this project as a scientific diagnostic rather than as the main objective.

Phase 7 intentionally used only:

- Logistic Regression as an interpretable linear baseline;
- RBF-SVM as one predefined nonlinear sensitivity model;
- cell-grouped cross-validation;
- paired time-reversal null models.

No broad model zoo, deep neural network, or result-driven hyperparameter search was used.

This limits overfitting and reduces the risk of selecting a model simply because it produces a desirable result.

---

## Repository Structure

```text
cell-irreversibility/
│
├── data/
├── envs/
├── figures/
├── instructions/
├── notebooks/
├── src/
├── tests/
├── env.yml
└── README.md
```

Current notebooks include:

```text
02_clean_spots_MSC01.ipynb
03_compute_steps_MSC01.ipynb
04_drift_validation_MSC01.ipynb
05_displacement_density_MSC01.ipynb
06_phase6_sequence_irreversibility_MSC01.ipynb
07_phase7_full_sequence_classification_MSC01.ipynb
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

### Phase 6

**[Standalone Phase 6 Handbook](instructions/Cell_Motility_Irreversibility_Phase6_Standalone_Handbook.pdf)**

The handbooks contain substantially more detail than this README, including:

- project history;
- code-level workflows;
- mathematical derivations;
- Python concepts;
- statistical reasoning;
- biophysical interpretation;
- validation procedures;
- sensitivity analyses;
- scientific decision points;
- reproducibility practices;
- and continuation notes for later project phases.

A standalone Phase 7 handbook will be added after final Phase 7 documentation is completed.

---

## Next Phase

Phase 8 will extend the temporal analysis beyond the present two-step framework.

Possible directions include:

- longer ordered sequences;
- temporal-memory observables;
- representations that separate directional motion from higher-order temporal structure;
- and methods that remain statistically appropriate for the available number of cells and trajectories.

The exact Phase 8 design will be selected before analysis rather than chosen in response to whether a particular method produces a positive result.

Entropy-production and TUR-style interpretations remain deferred until the trajectory analysis provides sufficient evidence to justify such a thermodynamic connection.

---

## Repository

https://github.com/AbolDpirate/cell-irreversibility

---

## Status

**Phase 7 completed. Phase 8 is next.**