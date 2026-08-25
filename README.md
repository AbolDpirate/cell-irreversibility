# Time Irreversibility in Cell Motility

A reproducible computational biophysics project exploring directional motion,
time-reversal asymmetry, and the limits of irreversibility inference from
cell-migration trajectories.

The project is also used as a structured learning project in Python,
scientific computing, statistics, biophysics, machine learning, testing,
Git/GitHub, and reproducible research.

> **Current status:** Phase 8B — Higher-Order Temporal Structure completed.
>
> **Next:** Phase 9 — nonequilibrium-thermodynamic interpretation and
> entropy-production-related analysis.

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

The analysis progressively moves from single-displacement directional
asymmetry toward explicitly ordered trajectory analysis:

```text
single-displacement directional asymmetry
        ↓
ordered two-step time reversal
        ↓
full two-step forward/reverse classification
        ↓
independent tracking-source validation
        ↓
higher-order three-step and four-step temporal structure
        ↓
nonequilibrium-thermodynamic interpretation
```

The present results remain exploratory.

They are **not** direct measurements of physical entropy production and do not
establish a thermodynamic arrow of time.

---

## Dataset

The current analysis uses sequence `01` of the **Fluo-C2DL-MSC** dataset from
the Cell Tracking Challenge (CTC).

The sequence contains rat mesenchymal stem cells imaged on a flat
polyacrylamide substrate.

Calibration used throughout the project:

- spatial resolution: `0.3 µm/pixel`
- temporal resolution: `20 min/frame`
- sequence length: `48 frames`

The original trajectory reconstruction used Fiji / TrackMate.

Phase 8A additionally introduced CTC `01_GT/TRA` annotations as an independent
**tracking reference**.

The CTC tracking annotations are not treated as absolute physical ground truth
for:

- biological motion;
- microscope or stage drift;
- irreversibility;
- or entropy production.

CTC and TrackMate are two trajectory reconstructions derived from the same
microscopy sequence and therefore are not independent biological datasets.

Agreement between them is interpreted as **tracking-source robustness**.

---

## Project Progress

Completed:

- **Phase 1:** reproducible Python/Conda environment
- **Phase 2:** TrackMate trajectory cleaning
- **Phase 3:** exact-lag displacement construction
- **Phase 4:** common-motion analysis and co-moving coordinates
- **Phase 5:** single-displacement spatial inversion asymmetry
- **Phase 6:** ordered two-step reduced-feature time-reversal analysis
- **Phase 7:** full two-step forward/reverse classification
- **Phase 8A:** CTC Gold-Reference Tracking Validation
- **Phase 8B:** higher-order three-step and four-step temporal-structure analysis

Next:

- **Phase 9:** nonequilibrium-thermodynamic interpretation,
  entropy-production-related quantities, and synthetic controls
- **Phase 10:** final synthesis, figures, documentation, code review,
  reproducibility audit, and project freeze

The repository is intentionally finite.

The planned endpoint is Phase 10, after which the project will be frozen as a
completed reproducible educational/scientific project rather than expanded
indefinitely.

---

## Exact-Lag Trajectory Analysis

Trajectory displacements are constructed by matching exact frame numbers
rather than using row offsets.

For lag `tau`, a displacement is retained only when both observations

```text
t
t + tau
```

exist for the same trajectory.

This prevents gaps in a track from being mistaken for valid consecutive
motion.

TrackMate exact-lag support used in the earlier displacement analyses:

| Lag | Physical time | Exact steps |
|---:|---:|---:|
| 1 frame | 20 min | 433 |
| 2 frames | 40 min | 398 |
| 4 frames | 80 min | 343 |

The authoritative exact-step implementation is maintained in:

```text
src/steps.py
```

---

## Raw and Co-Moving Representations

A population-level common-motion component was detected, particularly along
the x direction.

Both trajectory representations are retained:

```text
Raw trajectories
Co-moving trajectories
```

The co-moving representation subtracts frame-wise common population motion.

It is used as a sensitivity representation for testing whether trajectory
asymmetry survives removal of this collective translation.

The co-moving representation is **not** assumed to be an automatically more
physical estimate of cellular motion.

The physical origin of the population-level common-motion component remains
unresolved.

It must not be described as proven microscope or stage drift.

---

# Main Findings Through Phase 7

## Phase 5 — Single-Displacement Spatial Inversion Asymmetry

Single-displacement inversion asymmetry was quantified using

$$
D_{\mathrm{KL}}
\left[
p(\Delta)
\parallel
p(-\Delta)
\right].
$$

TrackMate baseline values were:

| `tau` | Raw | Co-moving |
|---:|---:|---:|
| 1 | 0.682 | 0.483 |
| 2 | 0.568 | 0.438 |
| 4 | 0.793 | 0.542 |

Common-motion subtraction reduced this directional asymmetry at all tested
lags.

This quantity is interpreted specifically as:

> **single-displacement spatial inversion asymmetry**

and not as full temporal irreversibility.

---

## Phase 6 — Ordered Two-Step Reduced-Feature Analysis

Exact consecutive two-step sequences were constructed as

$$
S=
(\mathbf d_1,\mathbf d_2).
$$

The correct time reversal is

$$
R(S)
=
(-\mathbf d_2,-\mathbf d_1).
$$

The analysis used time-odd reduced features derived from step-magnitude change
and turning geometry.

Neither the raw nor co-moving reduced-feature sequence asymmetry exceeded the
corresponding time-reversal-symmetrized null background.

The result therefore provided no null-calibrated evidence for two-step
sequence-order irreversibility in the tested reduced representation.

This non-detection does not prove that the underlying cellular dynamics are
time-reversible.

---

## Phase 7 — Full Two-Step Forward/Reverse Classification

Full two-step sequences were represented as

```text
(dx1, dy1, dx2, dy2)
```

and classified as forward versus exactly reversed.

TrackMate logistic regression produced approximately:

| Representation | Mean ROC AUC |
|---|---:|
| Raw | 0.622 |
| Co-moving | 0.504 |

The raw result exceeded its paired orientation-randomization null, whereas the
co-moving result did not.

Reversal-parity decomposition showed that almost all detectable raw classifier
information was reproduced by

$$
\mathbf d_1+\mathbf d_2
=
\mathbf r_{t+2}-\mathbf r_t,
$$

the net displacement over the two-step interval.

The reversal-even component was exactly at chance.

A fixed RBF-SVM sensitivity analysis did not reveal convincing additional
information in the complete sequence.

The Phase 7 conclusion was therefore:

> detectable raw time-direction information exists, but no convincing
> two-step sequence-order information was demonstrated beyond common
> directional motion.

---

# Phase 8A — CTC Gold-Reference Tracking Validation

Phase 8A tested whether the principal Phase 5–7 conclusions depended
critically on the TrackMate trajectory reconstruction.

CTC `01_GT/TRA` annotations produced:

```text
15 continuous reference tracks
428 track-frame observations
```

The reference trajectories were reconstructed from the CTC tracking masks and
validated before comparison with TrackMate.

---

## Tracking-Source Comparison

CTC and TrackMate were not identical trajectory reconstructions.

Differences included:

- TrackMate fragmentation and restart behavior;
- temporal coverage;
- track identity;
- localization differences.

Despite these differences, both sources reproduced a broad negative-x
population common-motion trend.

Approximate final cumulative x motion was:

```text
CTC reference: -18.74 µm
TrackMate:     -16.75 µm
```

The y component was less reproducible between tracking sources.

This supports tracking-source robustness of the broad x-directed population
motion but does not identify its physical origin.

---

## Phase 5 Replication

CTC exact-lag support was:

| Lag | Exact CTC steps |
|---:|---:|
| 1 | 413 |
| 2 | 398 |
| 4 | 369 |

CTC raw displacement asymmetry exceeded CTC co-moving asymmetry at all three
lags.

The raw-to-co-moving ordering survived:

- source-specific bandwidths;
- common-bandwidth sensitivity analysis;
- additional bandwidth multipliers;
- track-level paired bootstrap analysis.

Absolute asymmetry magnitudes differed between CTC and TrackMate.

Therefore:

> the direction of the common-motion effect is tracking-source robust, whereas
> the exact estimator magnitude is tracking-source sensitive.

The CTC measurements are not interpreted as showing that CTC trajectories are
“more irreversible.”

---

## Phase 6 Replication

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

---

## Phase 7 Replication

CTC contained only 15 independent track groups.

A pre-model fold audit showed that five-fold GroupKFold would leave too few
independent test tracks in some folds.

The CTC Phase 7 analysis therefore fixed:

```text
3-fold GroupKFold
```

before classifier performance was inspected.

CTC logistic mean ROC AUC was approximately:

| Representation | Mean ROC AUC |
|---|---:|
| Raw | 0.659 |
| Co-moving | 0.415 |

The raw result exceeded its paired orientation-randomization null.

The co-moving result did not.

Reversal decomposition again showed that the full classifier was essentially
equivalent to the net-displacement representation.

The reversal-even component remained exactly at chance.

A single fixed RBF-SVM sensitivity analysis produced the same qualitative
interpretation.

---

## Independent Whole-Image Motion Diagnostic

A limited whole-image phase-correlation analysis was performed as an
independent diagnostic of rigid image-content motion.

Approximate final cumulative x estimates were:

```text
Whole-image registration:  +5.79 µm
CTC trajectory common x:   -18.74 µm
TrackMate common x:         -16.75 µm
```

The whole-image estimate did not reproduce the strong negative-x trajectory
trend.

Therefore, the negative-x signal is robust across two trajectory
reconstructions but was not independently reproduced as a simple rigid
translation of the entire image.

This weakens a simple microscope/stage-drift interpretation.

It does not prove that the trajectory-level common motion is biological.

Its physical origin remains unresolved.

---

# Phase 8B — Higher-Order Temporal Structure

Phase 8B tested whether extending ordered trajectory analysis beyond the
two-step framework reveals forward-versus-reversed information that cannot be
explained by net directional displacement alone.

The Phase 8B analysis design was written and committed to Git **before**
higher-order classifier outcomes were inspected.

The pre-specified design is stored in:

```text
instructions/Phase8B_PreSpecified_Analysis_Design.md
```

The primary sequence length was three displacement steps.

A four-step extension was conditional on a pre-specified sample-support gate.

---

## Higher-Order Sequence Support

An exact three-step sequence requires four consecutive observations:

$$
\mathbf r_t,
\mathbf r_{t+1},
\mathbf r_{t+2},
\mathbf r_{t+3}.
$$

An exact four-step sequence requires five consecutive observations.

Observed support was:

| Tracking source | Exact 3-step sequences | 3-step groups | Exact 4-step sequences | 4-step groups |
|---|---:|---:|---:|---:|
| CTC reference | 383 | 14 | 369 | 14 |
| TrackMate | 357 | 28 | 329 | 26 |

The pre-specified four-step gate required:

```text
at least 200 exact four-step sequences
AND
at least 12 trajectory groups
```

Both tracking sources passed the gate.

Four-step analysis was therefore retained as the pre-specified secondary
extension.

---

# Three-Step Reversal-Parity Analysis

For

$$
S_3
=
(\mathbf d_1,\mathbf d_2,\mathbf d_3),
$$

the exact reversal is

$$
R(S_3)
=
(-\mathbf d_3,-\mathbf d_2,-\mathbf d_1).
$$

Three interpretable parity coordinates were defined.

Net reversal-odd displacement:

$$
\mathbf n
=
\mathbf d_1+\mathbf d_2+\mathbf d_3.
$$

Because of telescoping,

$$
\mathbf n
=
\mathbf r_{t+3}-\mathbf r_t.
$$

Internal reversal-odd structure:

$$
\mathbf q
=
\mathbf d_1
-
2\mathbf d_2
+
\mathbf d_3.
$$

Reversal-even structure:

$$
\mathbf e
=
\mathbf d_1-\mathbf d_3.
$$

Under reversal:

$$
\mathbf n\rightarrow-\mathbf n,
$$

$$
\mathbf q\rightarrow-\mathbf q,
$$

and

$$
\mathbf e\rightarrow\mathbf e.
$$

The transformation between the original six-dimensional sequence and
`(n, q, e)` was validated as invertible.

---

## Three-Step Feature Spaces

Five feature spaces were fixed before classifier performance was inspected:

```text
net_only             2D
internal_odd_only    2D
even_only            2D
odd_combined         4D
full_6d              6D
```

The central scientific comparison was:

$$
AUC_{\mathrm{odd\ combined}}
-
AUC_{\mathrm{net\ only}}.
$$

This explicitly tests whether internal reversal-odd structure adds detectable
time-direction information beyond net displacement.

---

# Four-Step Reversal-Parity Analysis

For

$$
S_4
=
(\mathbf d_1,\mathbf d_2,\mathbf d_3,\mathbf d_4),
$$

the exact reversal is

$$
R(S_4)
=
(-\mathbf d_4,-\mathbf d_3,-\mathbf d_2,-\mathbf d_1).
$$

The pre-specified reversal-odd coordinates were:

$$
\mathbf n_4
=
\mathbf d_1+\mathbf d_2+\mathbf d_3+\mathbf d_4,
$$

and

$$
\mathbf q_4
=
\mathbf d_1-\mathbf d_2-\mathbf d_3+\mathbf d_4.
$$

The reversal-even coordinates were:

$$
\mathbf e_{4,1}
=
\mathbf d_1-\mathbf d_4,
$$

and

$$
\mathbf e_{4,2}
=
\mathbf d_2-\mathbf d_3.
$$

Under reversal:

$$
\mathbf n_4\rightarrow-\mathbf n_4,
$$

$$
\mathbf q_4\rightarrow-\mathbf q_4,
$$

while both even coordinates remain unchanged.

The complete parity transformation was again validated as invertible.

---

## Four-Step Feature Spaces

The fixed feature hierarchy was:

```text
net_only             2D
internal_odd_only    2D
even_only            4D
odd_combined         4D
full_8d              8D
```

The complete reversal-even feature matrix is identical for forward and
reversed examples and therefore acts as a strict negative control.

---

# Phase 8B Classification Framework

All classifier analyses used trajectory-group-preserving cross-validation.

The pre-specified design was:

```text
3-fold GroupKFold
```

with trajectory identity as the grouping variable.

Observed four-step fold support was:

```text
CTC test groups:
4 / 5 / 5

TrackMate test groups:
8 / 9 / 9
```

All fold audits confirmed:

- zero train/test trajectory overlap;
- exact 50/50 forward/reversed class balance;
- multiple independent trajectory groups in every test fold.

---

## Primary Linear Model

The fixed primary classifier was:

```text
StandardScaler
      ↓
LogisticRegression(max_iter=2000)
```

No logistic-regression hyperparameter search was performed.

---

## Nonlinear Sensitivity Model

One pre-specified nonlinear sensitivity model was permitted:

```text
StandardScaler
      ↓
RBF-SVM(
    C=1.0,
    gamma="scale"
)
```

No hyperparameter optimization was performed.

No additional classifier was introduced after the results were observed.

---

## Paired Orientation-Randomization Null

Forward and reversed examples were maintained as exact sequence pairs.

For each null replicate, pair orientation was independently swapped with
probability 0.5.

The same orientation swap mask was used across compared feature spaces within
a replicate.

This allowed direct null calibration of both individual AUCs and paired model
contrasts.

The analysis used:

```text
200 null replicates
```

with fixed seeds specified before outcome inspection.

The minimum attainable empirical upper-tail p-value was therefore:

$$
\frac{1}{201}
\approx
0.004975.
$$

---

# Phase 8B Three-Step Results

## CTC Co-Moving Logistic Analysis

Mean ROC AUC:

| Feature space | AUC |
|---|---:|
| `net_only` | 0.474 |
| `internal_odd_only` | 0.516 |
| `even_only` | 0.500 |
| `odd_combined` | 0.483 |
| `full_6d` | 0.483 |

None exceeded the paired-null 95th percentile.

The primary contrast was approximately:

$$
AUC_{\mathrm{odd\ combined}}
-
AUC_{\mathrm{net}}
=
0.0086,
$$

and did not exceed its paired contrast-null reference.

The pre-specified primary higher-order evidence criterion was therefore not
satisfied.

---

## CTC Raw Logistic Analysis

Mean ROC AUC:

| Feature space | AUC |
|---|---:|
| `net_only` | 0.687 |
| `internal_odd_only` | 0.515 |
| `even_only` | 0.500 |
| `odd_combined` | 0.689 |
| `full_6d` | 0.689 |

The raw net, combined-odd, and full representations exceeded their paired
nulls.

However,

$$
AUC_{\mathrm{odd\ combined}}
-
AUC_{\mathrm{net}}
\approx
0.0025,
$$

and remained null-compatible.

Thus, the raw signal was explained by net displacement rather than detectable
internal three-step ordering.

---

## TrackMate Three-Step Replication

TrackMate reproduced the same qualitative pattern.

Co-moving logistic analysis remained below its paired null thresholds.

Raw logistic analysis showed:

```text
net_only       AUC ≈ 0.658
odd_combined   AUC ≈ 0.655
full_6d        AUC ≈ 0.655
```

but the internal-odd coordinate did not exceed its null, and neither the
odd-minus-net nor full-minus-net contrast was above null.

---

## Three-Step RBF Sensitivity

The fixed RBF-SVM sensitivity analysis produced the same interpretation.

Raw CTC and TrackMate trajectories contained detectable classifier
information, but net displacement alone reproduced the signal.

Co-moving analyses remained null-calibrated negative.

No nonlinear three-step model contrast provided convincing evidence beyond net
displacement.

---

# Phase 8B Four-Step Results

## CTC Co-Moving Logistic Analysis

Mean ROC AUC:

| Feature space | AUC |
|---|---:|
| `net_only` | 0.474 |
| `internal_odd_only` | 0.496 |
| `even_only` | 0.500 |
| `odd_combined` | 0.478 |
| `full_8d` | 0.478 |

None exceeded the paired-null 95th percentile.

The higher-order evidence criterion was not satisfied.

---

## CTC Raw Logistic Analysis

Mean ROC AUC:

```text
net_only       ≈ 0.720
odd_combined   ≈ 0.722
full_8d        ≈ 0.722
```

These raw classifiers exceeded their paired nulls.

However:

$$
AUC_{\mathrm{odd\ combined}}
-
AUC_{\mathrm{net}}
\approx
0.0021,
$$

which remained null-compatible.

The internal-odd coordinate alone did not exceed its null.

---

## TrackMate Four-Step Replication

TrackMate again reproduced the qualitative pattern.

Co-moving logistic classification remained null-calibrated negative.

Raw TrackMate produced approximately:

```text
net_only       AUC ≈ 0.665
odd_combined   AUC ≈ 0.663
full_8d        AUC ≈ 0.662
```

without detectable improvement beyond net displacement.

---

## Four-Step RBF Sensitivity

The fixed four-step RBF analysis again showed:

- no above-null co-moving higher-order signal;
- strong raw directional classifier information;
- no convincing internal-odd signal;
- no above-null odd-minus-net improvement.

No additional classifier or feature search was performed after this result.

---

# Integrated Phase 8B Result

The complete Phase 8B analysis contained:

$$
2
\times
2
\times
2
\times
2
=
16
$$

pre-specified combinations of:

- sequence length: three-step or four-step;
- tracking source: CTC or TrackMate;
- representation: raw or co-moving;
- classifier: logistic or fixed RBF-SVM.

The integrated result was:

| Diagnostic | Result |
|---|---:|
| Integrated analyses | 16 |
| Raw net-displacement signal above null | **8 / 8** |
| Co-moving net-displacement signal above null | **0 / 8** |
| Internal reversal-odd signal above null | **0 / 16** |
| Odd-minus-net contrast above null | **0 / 16** |
| Higher-order positive analyses | **0 / 16** |

This is the central Phase 8B result.

---

# Phase 8B Scientific Interpretation

Across both three-step and four-step sequence lengths, both tracking
reconstructions, and both classifiers:

1. raw trajectories consistently contained detectable forward/reverse
   information;
2. that information was already present in net displacement;
3. the internal reversal-odd coordinates never exceeded their null
   backgrounds;
4. adding internal reversal-odd structure never produced an above-null
   improvement over net displacement;
5. the complete sequence representations did not provide convincing
   incremental information beyond net displacement;
6. co-moving representations did not show convincing forward/reverse
   classification evidence.

Therefore:

> **Extending ordered trajectory analysis from two steps to three and four
> steps did not reveal detectable higher-order time-direction information
> beyond net directional displacement at the available data scale.**

This conclusion reproduced across CTC and TrackMate tracking reconstructions.

It therefore represents tracking-source robustness.

It does not constitute independent biological replication.

---

## What Phase 8B Does Not Show

The Phase 8B result does **not** establish that:

- the underlying cellular dynamics are time-reversible;
- microscopic detailed balance holds;
- physical entropy production is zero;
- the cell system is at thermodynamic equilibrium;
- all possible longer-timescale temporal information is absent.

The result is specifically a null-calibrated non-detection of higher-order
time-direction information under the tested sequence lengths, representations,
classifiers, and available sample size.

---

## Phase 8B Stopping Rule

The complete pre-specified Phase 8B analysis has been executed.

The project will not respond to the negative higher-order result by adding:

- five-step or longer sequences;
- arbitrary new features;
- classifier ensembles;
- random forests;
- gradient boosting;
- neural networks;
- deep learning;
- hyperparameter searches;
- post-hoc AUC inversion;
- alternative null models selected by performance.

Phase 8B is therefore scientifically complete.

---

# Current Scientific Interpretation

The strongest findings surviving Phases 5–8B are:

1. a broad directional population-motion component appears in both CTC and
   TrackMate trajectory reconstructions;
2. common-motion subtraction reduces single-displacement spatial inversion
   asymmetry;
3. reduced two-step sequence analysis remains null-calibrated negative;
4. raw two-step classifier information is reproducible;
5. raw classifier information is explained primarily by net displacement;
6. extending the sequence representation to three and four steps does not
   reveal detectable higher-order time-direction information beyond net
   displacement;
7. after common-motion subtraction, neither tracking reconstruction shows
   convincing forward/reverse classifier evidence under the tested higher-order
   models;
8. whole-image registration does not independently reproduce the large
   negative-x trajectory-level common motion.

The project currently does **not** justify:

- calling the common trajectory motion proven microscope/stage drift;
- claiming that the underlying cell dynamics are time-reversible;
- interpreting classifier AUC as a thermodynamic arrow of time;
- interpreting the present KL divergences directly as physical entropy
  production;
- or claiming zero entropy production from the negative higher-order results.

---

# Repository Structure

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

Raw and derived biological data remain local and are excluded from normal Git
tracking except for placeholder files required to preserve the repository
structure.

---

## Analysis Notebooks

Current analysis notebooks include:

```text
02_clean_spots_MSC01.ipynb
03_compute_steps_MSC01.ipynb
04_drift_validation_MSC01.ipynb
05_displacement_density_MSC01.ipynb
06_phase6_sequence_irreversibility_MSC01.ipynb
07_phase7_full_sequence_classification_MSC01.ipynb
08A_gold_reference_validation_MSC01.ipynb
08B_higher_order_temporal_structure_MSC01.ipynb
```

The completed Phase 8B notebook contains:

```text
117 cells
0 saved error outputs
```

and has been validated using a fresh-kernel full Run All.

---

## Reusable Source Modules

Current reusable analysis modules include:

```text
src/io.py
src/steps.py
src/density.py
src/metrics.py
src/plots.py
src/sequences.py
src/classification.py
```

`src/sequences.py` now includes reusable utilities for:

- exact arbitrary `k`-step sequence construction;
- exact sequence reversal;
- historical two-step analysis;
- three-step reversal-parity decomposition;
- four-step reversal-parity decomposition;
- exact parity reconstruction and validation.

---

# Environment and Tests

The project uses Python 3.11.

Create the environment with:

```bash
conda env create -f env.yml
conda activate cell-irreversibility
```

Run the automated test suite with:

```bash
python -m pytest -q
```

Current validated status:

```text
52 passed
```

The Phase 8B notebook has also been validated by:

1. restarting the Jupyter kernel;
2. running the complete notebook from beginning to end;
3. saving the completed notebook;
4. confirming zero saved error outputs.

---

# Detailed Documentation

Detailed scientific, mathematical, educational, and reproducibility records
are maintained in `instructions/`.

- [Master Handbook — Phases 0–4](instructions/Cell_Motility_Irreversibility_Master_Handbook.pdf)
- [Phase 5 Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase5_Standalone_Handbook.pdf)
- [Phase 6 Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase6_Standalone_Handbook.pdf)
- [Phase 7 Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase7_Standalone_Handbook.pdf)
- [Phase 8A Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase8A_Standalone_Handbook.pdf)
- [Phase 8B Pre-Specified Analysis Design](instructions/Phase8B_PreSpecified_Analysis_Design.md)
- [Phase 8B Standalone Handbook](instructions/Cell_Motility_Irreversibility_Phase8B_Standalone_Handbook.pdf)


The README is intentionally a project landing page.

Detailed cell-by-cell scientific and educational records belong in the
notebooks and standalone handbooks.

---

# Next Phase — Phase 9

Phase 9 will examine how the trajectory-level forward/reverse statistics
developed in the earlier phases relate to nonequilibrium thermodynamic
quantities.

The analysis will explicitly distinguish among:

- statistical time-direction information;
- path-space irreversibility;
- entropy-production estimators;
- entropy-production lower bounds;
- and physical thermodynamic entropy production.

The current displacement asymmetries and classifier AUCs will **not** simply be
renamed as entropy production.

Where necessary, synthetic or controlled stochastic-process examples will be
used to determine what candidate estimators can and cannot legitimately
measure.

Important questions for Phase 9 include:

- what forward/reversed path-probability ratio is actually estimable from the
  available trajectory data;
- what coarse-graining does to entropy-production inference;
- whether any path-space statistic admits a defensible lower-bound
  interpretation;
- how finite sampling affects such estimates;
- whether the estimator behaves correctly on synthetic reversible and
  irreversible processes with known structure;
- and which conclusions can legitimately be transferred to the experimental
  cell trajectories.

Phase 9 will preserve the distinction between statistical irreversibility and
physical thermodynamic interpretation.

---

# Final Planned Phase — Phase 10

Phase 10 will close the project.

It will include:

- final scientific synthesis;
- publication-quality figures;
- final numerical tables;
- cross-phase interpretation;
- limitations;
- final code review;
- cleanup of duplicated or obsolete analysis code where appropriate;
- complete test and reproducibility audit;
- final README;
- final Master Handbook update;
- final project documentation;
- Git/GitHub cleanup;
- release-style project endpoint;
- and repository freeze.

No open-ended Phase 11 expansion is currently planned.

---

# Repository

https://github.com/AbolDpirate/cell-irreversibility

---

# Status

**Phase 8B completed scientifically and validated reproducibly.**

**Phase 9 — Nonequilibrium Thermodynamic Interpretation is next.**