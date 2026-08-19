# Time Irreversibility in Cell Motility

A reproducible computational biophysics project exploring directional motion,
time-reversal asymmetry, and the limits of irreversibility inference from
cell-migration trajectories.

The project is also used as a structured learning project in Python,
scientific computing, statistics, biophysics, machine learning, testing,
Git/GitHub, and reproducible research.

> **Current status:** Phase 8A — CTC Gold-Reference Tracking Validation completed.
>
> **Next:** Phase 8B — higher-order temporal structure.

---

## Scientific Aim

Living cells are active systems that consume energy and operate away from
thermodynamic equilibrium.

This project asks whether experimentally measured cell trajectories contain
statistical information that distinguishes forward dynamics from appropriately
time-reversed dynamics.

For a trajectory,

$$
\Delta_\tau \mathbf{r}(t)
=
\mathbf{r}(t+\tau)-\mathbf{r}(t).
$$

The project progresses from single-displacement directional asymmetry toward
explicitly ordered trajectory analysis.

Current results are exploratory. They are **not** direct measurements of
entropy production and do not establish a thermodynamic arrow of time.

---

## Dataset

The current analysis uses sequence `01` of the **Fluo-C2DL-MSC** dataset from
the Cell Tracking Challenge (CTC).

Calibration:

- spatial resolution: `0.3 µm/pixel`
- temporal resolution: `20 min/frame`
- sequence length: `48 frames`

The original trajectory reconstruction used Fiji / TrackMate.

Phase 8A additionally used CTC `01_GT/TRA` annotations as an independent
**tracking reference**.

These annotations are not treated as physical ground truth for microscope
drift, biological motion, irreversibility, or entropy production.

---

## Project Progress

Completed:

- **Phase 1:** reproducible Python/Conda setup
- **Phase 2:** TrackMate trajectory cleaning
- **Phase 3:** exact-lag displacement construction
- **Phase 4:** common-motion analysis and co-moving coordinates
- **Phase 5:** single-displacement spatial inversion asymmetry
- **Phase 6:** ordered two-step reduced-feature time-reversal analysis
- **Phase 7:** full two-step forward/reverse classification
- **Phase 8A:** independent CTC tracking-reference validation

Next:

- **Phase 8B:** higher-order temporal structure
- **Phase 9:** nonequilibrium-thermodynamic interpretation and synthetic controls
- **Phase 10:** final synthesis, figures, documentation, code review, and project freeze

---

## Exact-Lag Trajectory Analysis

The authoritative displacement implementation matches observations using exact
frame numbers rather than row offsets.

TrackMate exact-lag support:

| Lag | Physical time | Exact steps |
|---:|---:|---:|
| 1 frame | 20 min | 433 |
| 2 frames | 40 min | 398 |
| 4 frames | 80 min | 343 |

Implementation:

```text
src/steps.py
```

---

## Raw and Co-Moving Representations

A population-level common-motion component was detected, particularly along
the x direction.

Both representations are retained:

```text
Raw trajectories
Co-moving trajectories
```

The co-moving representation subtracts frame-wise common population motion.

It is a sensitivity representation, not an automatically superior estimate of
true cellular motion.

The physical origin of the common-motion component remains unresolved.

---

## Main Findings Through Phase 7

### Phase 5

Single-displacement inversion asymmetry was quantified using

$$
D_{\mathrm{KL}}
\left[
p(\Delta)
\parallel
p(-\Delta)
\right].
$$

TrackMate baseline values:

| `tau` | Raw | Co-moving |
|---:|---:|---:|
| 1 | 0.682 | 0.483 |
| 2 | 0.568 | 0.438 |
| 4 | 0.793 | 0.542 |

This is interpreted as **single-displacement spatial inversion asymmetry**, not
full temporal irreversibility.

### Phase 6

Exact consecutive two-step sequences were analyzed using the correct reversal

$$
(\mathbf d_1,\mathbf d_2)
\rightarrow
(-\mathbf d_2,-\mathbf d_1).
$$

Neither raw nor co-moving reduced-feature sequence asymmetry exceeded the
time-reversal-symmetrized null background.

### Phase 7

Forward-versus-reversed full-sequence classification found a detectable raw
linear signal but little co-moving signal.

Reversal decomposition showed that almost all of the raw classifier
information was explained by

$$
\mathbf d_1+\mathbf d_2
=
\mathbf r_{t+2}-\mathbf r_t,
$$

the net two-frame displacement.

The reversal-even component was at chance, and one pre-specified RBF-SVM
sensitivity analysis did not reveal convincing additional information in the
complete two-step sequence.

Therefore, the two-step analyses did **not** provide convincing evidence for
sequence-order irreversibility beyond common directional motion.

---

## Phase 8A — CTC Gold-Reference Tracking Validation

Phase 8A tested whether the principal Phase 5–7 conclusions depended
critically on the TrackMate trajectory reconstruction.

CTC `01_GT/TRA` produced:

```text
15 continuous reference tracks
428 track-frame observations
```

### Tracking-source comparison

CTC and TrackMate were not identical reconstructions. Differences included
fragmentation/restart behavior, temporal coverage, and localization.

Despite these differences, both trajectory sources reproduced a broad
negative-x population common-motion trend.

Approximate final cumulative x motion:

```text
CTC reference: -18.74 µm
TrackMate:     -16.75 µm
```

The y component was less reproducible.

### Phase 5 replication

CTC raw displacement asymmetry exceeded CTC co-moving asymmetry at
`tau = 1, 2, 4`.

The raw-to-co-moving ordering survived bandwidth sensitivity and paired
track-level resampling.

Thus, the contribution of common population motion to single-displacement
spatial asymmetry is tracking-source robust, although exact metric magnitudes
are tracking-source sensitive.

### Phase 6 replication

CTC produced:

```text
398 exact two-step sequences
392 paired valid-turn sequences
15 reference tracks
```

Neither raw nor co-moving reduced-feature asymmetry exceeded its
time-reversal-symmetrized null 95th percentile.

The central Phase 6 null-calibrated non-detection therefore reproduced across
tracking sources.

### Phase 7 replication

Because CTC contains only 15 independent track groups, a pre-model audit
rejected five-fold GroupKFold and fixed the analysis to three-fold GroupKFold
before classifier performance was inspected.

CTC logistic mean ROC AUC:

| Representation | Mean ROC AUC |
|---|---:|
| Raw | 0.659 |
| Co-moving | 0.415 |

The raw result exceeded its paired orientation-randomization null; the
co-moving result did not.

Reversal decomposition again showed that the full raw classifier was
essentially identical to the odd net-displacement classifier, while the even
component was exactly at chance.

A single fixed RBF-SVM sensitivity analysis reached the same interpretation:
the complete two-step sequence did not provide convincing information beyond
net directional displacement.

### Independent image-motion diagnostic

A limited whole-image phase-correlation analysis did **not** reproduce the
strong negative-x trajectory trend.

Approximate final cumulative x estimates:

```text
Whole-image registration:  +5.79 µm
CTC trajectory common x:   -18.74 µm
TrackMate common x:         -16.75 µm
```

Therefore, the negative-x signal is robust across two trajectory
reconstructions but was not independently reproduced as a simple rigid
whole-image translation.

This weakens a simple microscope/stage-drift interpretation but does not prove
that the signal is biological.

Its physical origin remains unresolved.

---

## Current Scientific Interpretation

The strongest findings surviving Phase 8A are:

1. broad negative-x directional motion appears in both TrackMate and CTC
   trajectory reconstructions;
2. common-motion subtraction reduces single-displacement spatial asymmetry;
3. the reduced two-step Phase 6 analysis remains null-calibrated negative;
4. raw forward/reverse classifier information is reproducible;
5. that classifier information is explained almost entirely by net
   displacement rather than richer two-step ordering.

The project currently does **not** justify:

- calling the common motion proven microscope/stage drift;
- claiming the underlying dynamics are time-reversible;
- interpreting classifier AUC as a thermodynamic arrow of time;
- or estimating physical entropy production from the present statistics.

---

## Repository Structure

```text
cell-irreversibility/
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

Current analysis notebooks include:

```text
02_clean_spots_MSC01.ipynb
03_compute_steps_MSC01.ipynb
04_drift_validation_MSC01.ipynb
05_displacement_density_MSC01.ipynb
06_phase6_sequence_irreversibility_MSC01.ipynb
07_phase7_full_sequence_classification_MSC01.ipynb
08A_gold_reference_validation_MSC01.ipynb
```

Reusable source modules include:

```text
src/steps.py
src/density.py
src/metrics.py
src/sequences.py
src/classification.py
```

---

## Environment and Tests

The project uses Python 3.11.

Create the environment with:

```bash
conda env create -f env.yml
conda activate cell-irreversibility
```

Run tests with:

```bash
python -m pytest -q
```

Current validated status:

```text
36 passed
```

The Phase 8A notebook has also been validated by restarting the kernel and
running the complete notebook from beginning to end.

---

## Detailed Documentation

Detailed scientific, mathematical, educational, and reproducibility records
are maintained in `instructions/`.

- [Master Handbook — Phases 0–4](instructions/Cell_Motility_Irreversibility_Master_Handbook.pdf)
- [Phase 5 Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase5_Standalone_Handbook.pdf)
- [Phase 6 Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase6_Standalone_Handbook.pdf)
- [Phase 7 Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase7_Standalone_Handbook.pdf)
- [Phase 8A Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase8A_Standalone_Handbook.pdf)

The README is intentionally a project landing page; detailed cell-by-cell
analysis belongs in the notebooks and standalone handbooks.

---

## Next Phase — Phase 8B

Phase 8B will test higher-order temporal structure using longer ordered
sequences, likely three-step and, if statistically defensible, four-step
representations.

The analysis design will be fixed before inspecting outcome statistics.

Additional classifiers will not be added merely to search for a positive
result.

---

## Repository

https://github.com/AbolDpirate/cell-irreversibility

---

## Status

**Phase 8A completed and validated. Phase 8B is next.**