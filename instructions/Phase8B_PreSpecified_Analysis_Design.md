# Phase 8B — Pre-Specified Analysis Design

## Higher-Order Temporal Structure in Cell-Motility Trajectories

**Project:** Time Irreversibility in Cell Motility  
**Phase:** 8B  
**Status:** Pre-specified before Phase 8B outcome inspection  
**Primary purpose:** Test whether longer ordered displacement sequences contain
forward-versus-reversed information beyond net directional displacement.

---

## 1. Scientific Motivation

Phases 6, 7, and 8A established an important distinction.

Two-step trajectory sequences can contain detectable forward-versus-reversed
information in the raw representation. However, reversal-parity decomposition
showed that this signal is almost entirely reproduced by the net displacement

\[
\mathbf d_1+\mathbf d_2
=
\mathbf r_{t+2}-\mathbf r_t.
\]

The complete two-step sequence did not provide convincing additional
time-direction information beyond this directional component.

Phase 8B therefore asks a more demanding question:

> Do longer ordered displacement sequences contain detectable temporal
> information that cannot be explained by net directional displacement alone?

Phase 8B is not designed merely to test whether a classifier can distinguish
forward from reversed trajectories.

The central target is **higher-order temporal structure beyond net motion**.

---

## 2. Scope

Phase 8B remains a limited extension of the existing single-dataset project.

It will not become:

- a multi-dataset benchmark;
- a classifier search;
- a hyperparameter-optimization project;
- a broad feature-engineering exercise;
- an entropy-production analysis;
- or a publication-scale benchmarking study.

The primary analysis is based on exact three-step sequences.

A four-step extension is conditional on a pre-specified sample-support gate.

---

## 3. Tracking Sources

### Primary tracking source

CTC `01_GT/TRA` reference tracking.

The CTC annotations are treated as a tracking reference, not as absolute
physical ground truth.

### Tracking-source sensitivity analysis

TrackMate cleaned trajectories.

CTC and TrackMate originate from the same microscopy sequence and therefore
must not be interpreted as independent biological replication.

Agreement between them represents **tracking-source robustness**.

---

## 4. Raw and Co-Moving Representations

Both trajectory representations will be retained:

1. raw;
2. co-moving.

The raw representation preserves population-level directional motion.

The co-moving representation subtracts the previously defined frame-wise
common population translation.

The co-moving representation is the primary interpretive representation for
the question of higher-order temporal structure beyond common directional
motion.

It is not assumed to be the true physical trajectory.

---

# 5. Exact Three-Step Sequences

The primary Phase 8B sequence is

\[
S_3
=
(\mathbf d_1,\mathbf d_2,\mathbf d_3),
\]

constructed from four exactly consecutive observations:

\[
\mathbf r_t,\;
\mathbf r_{t+1},\;
\mathbf r_{t+2},\;
\mathbf r_{t+3}.
\]

The displacements are

\[
\mathbf d_1
=
\mathbf r_{t+1}-\mathbf r_t,
\]

\[
\mathbf d_2
=
\mathbf r_{t+2}-\mathbf r_{t+1},
\]

and

\[
\mathbf d_3
=
\mathbf r_{t+3}-\mathbf r_{t+2}.
\]

Missing intermediate frames are not permitted.

Row adjacency is not sufficient.

Exact frame continuity must be explicitly verified.

---

## 6. Exact Three-Step Time Reversal

The correct time reversal is

\[
R(S_3)
=
(-\mathbf d_3,-\mathbf d_2,-\mathbf d_1).
\]

The reversal implementation must satisfy the involution property

\[
R(R(S_3))=S_3.
\]

This property will be protected by automated tests.

---

# 7. Three-Step Reversal-Parity Coordinates

The full three-step sequence contains six scalar coordinates.

For physical interpretation, it will also be represented using three
two-dimensional vector coordinates.

## 7.1 Net reversal-odd coordinate

Define

\[
\mathbf n
=
\mathbf d_1+\mathbf d_2+\mathbf d_3.
\]

This is exactly

\[
\mathbf n
=
\mathbf r_{t+3}-\mathbf r_t,
\]

the net displacement across the complete three-step interval.

Under time reversal,

\[
\mathbf n\rightarrow-\mathbf n.
\]

This coordinate therefore captures directional motion but does not by itself
demonstrate higher-order temporal structure.

---

## 7.2 Internal reversal-odd coordinate

Define

\[
\mathbf q
=
\mathbf d_1-2\mathbf d_2+\mathbf d_3.
\]

Under time reversal,

\[
\mathbf q\rightarrow-\mathbf q.
\]

Unlike the net coordinate, \(\mathbf q\) depends on the internal distribution
of displacement across the three ordered steps.

It is therefore an interpretable candidate for higher-order time-odd
structure.

---

## 7.3 Reversal-even coordinate

Define

\[
\mathbf e
=
\mathbf d_1-\mathbf d_3.
\]

Under time reversal,

\[
\mathbf e\rightarrow\mathbf e.
\]

Because this coordinate is reversal invariant, an even-only
forward-versus-reversed classifier should contain no class information.

This provides an internal sanity check.

---

## 7.4 Invertibility

The transformation

\[
(\mathbf d_1,\mathbf d_2,\mathbf d_3)
\leftrightarrow
(\mathbf n,\mathbf q,\mathbf e)
\]

is an invertible linear transformation.

Therefore, the parity-coordinate representation does not discard information
from the complete six-dimensional sequence.

---

# 8. Pre-Specified Three-Step Feature Spaces

Exactly five feature spaces will be evaluated.

### 1. Net only

\[
\mathbf n
\]

Dimension: 2.

Purpose: quantify time-direction information explained by net displacement.

### 2. Internal odd only

\[
\mathbf q
\]

Dimension: 2.

Purpose: test internal reversal-odd structure without net displacement.

### 3. Even only

\[
\mathbf e
\]

Dimension: 2.

Purpose: sanity control.

Expected behavior: chance-level forward/reverse discrimination.

### 4. Combined odd coordinates

\[
[\mathbf n,\mathbf q]
\]

Dimension: 4.

Purpose: determine whether internal time-odd structure adds information beyond
net displacement.

### 5. Full sequence

Equivalent to either

\[
[\mathbf d_1,\mathbf d_2,\mathbf d_3]
\]

or the invertible representation

\[
[\mathbf n,\mathbf q,\mathbf e].
\]

Dimension: 6.

Purpose: test whether the complete sequence contains additional information
beyond the reversal-odd subspace.

No additional data-driven feature spaces will be introduced after outcome
inspection.

---

# 9. Classification Labels

For every observed forward sequence,

\[
S,
\]

its exact reversed partner

\[
R(S)
\]

will be generated.

Labels are fixed as:

```text
forward = 1
reversed = 0
```

AUC values below 0.5 will be retained as observed.

They will not be post-hoc transformed to `1 - AUC`.

Class labels will not be flipped after results are observed.

---

# 10. Biological Grouping and Cross-Validation

Sequences originating from the same trajectory are statistically dependent.

All classification analyses therefore use trajectory-group-preserving
cross-validation.

## CTC

```text
3-fold GroupKFold
group = CTC reference track
```

## TrackMate sensitivity analysis

```text
3-fold GroupKFold
group = TrackMate cell_id
```

Three folds are fixed in advance for Phase 8B.

The same fold design will be used across feature spaces and models within each
tracking source.

Train/test trajectory overlap must equal zero.

---

# 11. Primary Linear Classifier

The primary classifier is fixed as:

```text
StandardScaler
      ↓
LogisticRegression(max_iter=2000)
```

Scaling must occur inside the scikit-learn pipeline so that preprocessing is
estimated only from training groups.

No logistic-regression hyperparameter search will be performed.

---

# 12. Nonlinear Sensitivity Model

One nonlinear sensitivity model is permitted:

```text
StandardScaler
      ↓
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
```

No hyperparameter optimization will be performed.

No additional classifier will be introduced based on observed Phase 8B
performance.

---

# 13. Paired Orientation-Randomization Null

The classifier null will preserve the observed forward/reversed pairing.

For every sequence pair

\[
(S,R(S)),
\]

a Bernoulli probability of 0.5 determines whether the pair orientation is
swapped.

Within a given null replicate, the same orientation swap mask must be used
across all feature spaces being compared.

This preserves paired model contrasts.

The same conceptual null is used for raw and co-moving analyses.

Number of null replicates:

```text
200
```

Fixed Phase 8B random seeds:

```text
Logistic null seed = 2029
RBF null seed      = 2030
```

The empirical upper-tail p-value is

\[
p_{\mathrm{emp}}
=
\frac{
1+\#(T_{\mathrm{null}}\geq T_{\mathrm{obs}})
}{
1+N_{\mathrm{null}}
}.
\]

With 200 null replicates, the minimum attainable empirical p-value is

\[
1/201
\approx
0.004975.
\]

---

# 14. Primary Phase 8B Scientific Test

The principal inferential question is evaluated in the:

```text
CTC
co-moving
logistic-regression
three-step
```

analysis.

The primary comparison is:

\[
AUC_{\mathrm{odd\ combined}}
-
AUC_{\mathrm{net\ only}}.
\]

Evidence for higher-order reversal-odd structure requires both:

1. the combined-odd classifier to exceed its paired null 95th percentile; and
2. the observed

\[
AUC_{\mathrm{odd\ combined}}
-
AUC_{\mathrm{net\ only}}
\]

contrast to exceed the corresponding paired contrast-null 95th percentile.

This prevents a strong net-displacement signal from being misinterpreted as
higher-order temporal information.

---

# 15. Supporting Three-Step Diagnostics

The following are pre-specified supporting analyses.

## Internal odd only

The \(\mathbf q\)-only classifier will be compared with its paired null.

An above-null result supports the presence of internal reversal-odd
information.

## Even only

The \(\mathbf e\)-only representation should produce chance-level
classification because forward and reversed even coordinates are identical.

A substantial departure from this expectation will trigger a pipeline audit
rather than a scientific interpretation.

## Full versus combined odd

The contrast

\[
AUC_{\mathrm{full}}
-
AUC_{\mathrm{odd\ combined}}
\]

asks whether reversal-even information contributes through interactions not
captured by the linear odd subspace.

This is secondary to the primary odd-combined versus net comparison.

---

# 16. Interpretation Hierarchy

The following patterns are distinguished in advance.

## Pattern A — directional signal only

If

```text
net-only is above null
full is above null
odd-combined ≈ net-only
```

then the result is interpreted as directional information without convincing
higher-order temporal structure.

## Pattern B — internal higher-order odd information

If the primary CTC co-moving odd-combined-minus-net contrast exceeds its paired
null while the combined-odd classifier itself is above null, this constitutes
evidence for detectable three-step temporal information beyond net
displacement in the tested representation.

## Pattern C — nonlinear full-sequence contribution

If the fixed RBF full-sequence model exceeds its null and significantly
outperforms the appropriate lower-dimensional comparison under the paired
contrast null, this is treated as secondary evidence of nonlinear
higher-order sequence information.

## Pattern D — no detectable higher-order signal

If the above criteria are not met, the Phase 8B conclusion will be that the
available three-step analysis did not detect temporal information beyond net
directional motion at the available data scale.

A negative result will not trigger additional model or feature searches.

---

# 17. Tracking-Source Sensitivity

After completing the pre-specified CTC primary analysis, the same three-step
logic will be applied to TrackMate trajectories.

Tracking-source agreement strengthens robustness.

Disagreement will be reported directly.

CTC and TrackMate results will not be treated as statistically independent
biological replications.

---

# 18. Conditional Four-Step Extension

Four-step analysis is secondary and conditional.

A four-step sequence is

\[
S_4
=
(\mathbf d_1,\mathbf d_2,\mathbf d_3,\mathbf d_4)
\]

with reversal

\[
R(S_4)
=
(-\mathbf d_4,-\mathbf d_3,-\mathbf d_2,-\mathbf d_1).
\]

Four-step inferential analysis will be performed for a tracking source only if
the support audit shows:

```text
at least 200 exact four-step sequences
AND
at least 12 trajectory groups
```

These thresholds are fixed before the support counts are inspected.

If either condition fails, four-step inferential analysis will not be
performed for that tracking source.

The thresholds will not be lowered after observing the data.

---

# 19. Four-Step Parity Coordinates

If the four-step support gate passes, the pre-specified coordinates are:

## Net reversal-odd coordinate

\[
\mathbf n_4
=
\mathbf d_1+\mathbf d_2+\mathbf d_3+\mathbf d_4.
\]

## Internal reversal-odd coordinate

\[
\mathbf q_4
=
\mathbf d_1-\mathbf d_2-\mathbf d_3+\mathbf d_4.
\]

Both satisfy

\[
\mathbf n_4\rightarrow-\mathbf n_4
\]

and

\[
\mathbf q_4\rightarrow-\mathbf q_4.
\]

Two reversal-even coordinates are

\[
\mathbf e_{4,1}
=
\mathbf d_1-\mathbf d_4
\]

and

\[
\mathbf e_{4,2}
=
\mathbf d_2-\mathbf d_3.
\]

These remain unchanged under reversal.

The four-step feature hierarchy will mirror the three-step analysis:

- net only;
- internal odd only;
- combined odd;
- even only;
- full sequence.

No alternative decomposition will be selected after outcome inspection.

---

# 20. Explicitly Excluded Adaptive Analyses

Phase 8B will not introduce, in response to observed outcomes:

- alternative sequence lengths beyond the pre-specified three- and conditional
  four-step analyses;
- arbitrary new hand-designed features;
- alternative labels;
- post-hoc AUC inversion;
- random forest;
- gradient boosting;
- XGBoost;
- neural networks;
- deep learning;
- classifier ensembles;
- broad hyperparameter tuning;
- feature selection based on outcome;
- repeated cross-validation variants selected by performance;
- additional null definitions selected by performance.

Any later methodological extension would require a new explicitly documented
analysis phase rather than being silently added to Phase 8B.

---

# 21. Thermodynamic Boundary

Phase 8B remains a trajectory-statistics phase.

The following terms will not be treated as equivalent:

- forward/reverse classification;
- time-direction information;
- path-space asymmetry;
- physical entropy production.

Entropy-production interpretation is deferred to Phase 9.

Phase 8B may identify statistical time-direction information but does not by
itself establish a thermodynamic entropy-production rate.

---

# 22. Stopping Rule

Phase 8B ends after:

1. exact three-step support audit;
2. three-step construction and validation;
3. parity decomposition;
4. pre-specified CTC logistic analysis;
5. paired null and model-contrast analysis;
6. fixed RBF sensitivity analysis;
7. TrackMate tracking-source sensitivity;
8. conditional four-step analysis if the pre-specified support gate passes;
9. integrated scientific synthesis;
10. tests, documentation, and repository validation.

No additional classifier or feature search will be added because of a negative
result.

---

# 23. Phase 8B Success Definition

A positive higher-order result is **not** defined merely as successful
forward/reverse classification.

The central Phase 8B claim requires evidence that longer sequences contain
time-direction information that cannot be explained by net displacement alone,
with primary emphasis on the CTC co-moving representation.

A scientifically valid negative endpoint is:

> Extending the analysis from two-step to three-step, and conditionally
> four-step, ordered trajectories did not reveal detectable temporal-order
> information beyond directional motion at the available data scale.

Both positive and negative outcomes complete the phase.

---

# 24. Reproducibility Commitments

Before inferential results are inspected:

- this design document will be committed to Git;
- the Phase 8B branch will be created;
- the primary and secondary analyses will be fixed;
- sample-support thresholds will be fixed;
- null-replicate counts and seeds will be fixed;
- classifier choices will be fixed;
- the stopping rule will be fixed.

The first data-dependent Phase 8B computation after this commit will be a
sequence-support audit only.

No classifier performance or null result will be inspected before the design
commit.

---

## Frozen Phase 8B Parameters

```text
Primary sequence length:
3 steps

Conditional secondary sequence length:
4 steps

Primary tracking source:
CTC reference

Tracking-source sensitivity:
TrackMate

Representations:
raw and co-moving

Primary interpretive representation:
CTC co-moving

Cross-validation:
3-fold GroupKFold

Primary classifier:
StandardScaler -> LogisticRegression(max_iter=2000)

Nonlinear sensitivity:
StandardScaler -> RBF-SVM(C=1.0, gamma="scale")

Null:
paired orientation randomization

Null replicates:
200

Logistic null seed:
2029

RBF null seed:
2030

Four-step minimum exact sequences:
200

Four-step minimum trajectory groups:
12

Primary higher-order contrast:
combined odd AUC - net-only AUC

Primary positive-evidence requirement:
combined odd above paired null
AND
combined-odd-minus-net above paired contrast null

No post-hoc AUC inversion:
yes

No classifier zoo:
yes

No outcome-driven feature search:
yes
```

---

## Immediate Next Step After Design Commit

Perform only a sample-support audit for exact three-step and exact four-step
sequences in CTC and TrackMate trajectories.

No classification or inferential outcome statistic will be computed until the
support audit is complete and the four-step eligibility decision has been
recorded.