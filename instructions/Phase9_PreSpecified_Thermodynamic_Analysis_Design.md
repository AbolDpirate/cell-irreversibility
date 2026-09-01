# Phase 9 — Pre-Specified Thermodynamic Analysis Design

**Project:** Time Irreversibility in Cell Motility
**Phase:** 9 — Nonequilibrium Thermodynamic Interpretation and Entropy-Production-Related Analysis
**Pre-specification date:** 1 September 2026
**Status:** Analysis design locked before inspection of any Phase 9 scientific output

---

## 1. Purpose

Phase 9 develops a scientifically controlled bridge between the statistical time-reversal analyses completed in Phases 5–8B and nonequilibrium stochastic thermodynamics.

The phase has two primary goals:

1. build and validate entropy-production analysis on synthetic stochastic systems for which the thermodynamic answer is known;
2. apply a conservative path-space irreversibility estimator to the experimental MSC01 cell trajectories without overstating its thermodynamic meaning.

This phase is intentionally limited in scope.

It is not intended to become a general stochastic-thermodynamics methods project, a large model-comparison exercise, or a search for a positive experimental result.

---

## 2. Scientific Interpretation Boundary

The following quantities and concepts must remain distinct throughout Phase 9:

* directional asymmetry;
* forward-versus-reversed classification;
* path-space KL divergence;
* coarse-grained path-space irreversibility;
* entropy-production-related lower bounds;
* physical thermodynamic entropy production.

These concepts are related, but they are not interchangeable.

Physical entropy production will be named only when the stochastic model, observed variables, and thermodynamic assumptions justify that interpretation.

The MSC01 data contain coarse-grained two-dimensional cell-centroid trajectories. They do not directly observe the molecular, chemical, or mechanical degrees of freedom responsible for the total dissipation of a living cell.

Therefore, trajectory-based irreversibility in MSC01 must not automatically be interpreted as total cellular entropy production.

---

## 3. Synthetic Benchmark

The primary synthetic benchmark is a two-dimensional rotational Ornstein–Uhlenbeck process:

$$
dX = A X\,dt + \sqrt{2D}\,dW,
$$

with

$$
A=-kI+\omega R,
$$

where \(I\) is the two-dimensional identity matrix and \(R\) is the planar rotation generator.

The baseline parameters are fixed as:

```text
k = 1
D = 1
```

The synthetic random seed is fixed as:

```text
2031
```

The Ornstein–Uhlenbeck system is a methodological benchmark. It is not intended as a biological model of mesenchymal stem-cell motility.

---

## 4. Equilibrium Synthetic Control

The equilibrium control is defined by:

```text
omega = 0
```

Under this condition there is no rotational steady-state probability current.

For the chosen model, the steady-state physical entropy-production rate is:

$$
\sigma = 0.
$$

The equilibrium benchmark must therefore provide a negative control for the thermodynamic and path-space estimators developed in this phase.

---

## 5. Nonequilibrium Rotational Benchmark

The nonequilibrium condition is defined by:

```text
omega = 1
```

This produces a rotational steady-state probability current.

For the isotropic continuous-time model,

$$
\sigma
=
\frac{2\omega^2}{k}.
$$

With

```text
k = 1
omega = 1
```

the analytical physical entropy-production rate is:

$$
\sigma = 2
$$

in units of \(k_B\) per simulation-time unit.

The sign of \(\omega\) changes the direction of rotation but does not change the entropy-production rate because the analytical rate depends on \(\omega^2\).

---

## 6. Synthetic Validation Goals

The synthetic analysis must demonstrate the following:

1. the equilibrium model has zero analytical entropy-production rate;
2. the rotational nonequilibrium model has positive analytical entropy-production rate;
3. forward-versus-reversed path-probability ratios recover or approach the known entropy production within sampling and discretization error;
4. partial observation or coarse-graining can reduce observable time-reversal information;
5. weak observed trajectory irreversibility does not imply weak microscopic dissipation.

The numerical validation tolerance must be chosen on numerical and statistical grounds and must not be changed after observing an inconvenient result merely to force agreement.

---

## 7. Path-Probability Ratio

The central path statistic is

$$
\log
\frac{P[\Gamma]}
{P[\Gamma^R]},
$$

where \(\Gamma\) denotes a forward path and \(\Gamma^R\) denotes its correctly time-reversed path.

Under appropriate stochastic-thermodynamic assumptions, the expectation of this log path-probability ratio is related to entropy production.

For the synthetic model, where the stochastic dynamics are explicitly defined, this relationship can be tested directly.

For the experimental MSC01 trajectories, the corresponding forward/reverse path comparison measures observable path-space irreversibility and must not automatically be interpreted as total physical cellular entropy production.

---

## 8. Coarse-Graining Demonstration

The synthetic benchmark will include at least two observation conditions:

```text
full two-dimensional observation
one-coordinate observation with the other coordinate hidden
```

The purpose is to demonstrate that hidden variables can reduce observable time-reversal information.

This analysis is intended to illustrate information loss under coarse-graining.

It does not attempt to reconstruct unobserved cellular molecular degrees of freedom.

---

## 9. Experimental Variational KL Lower Bound

The experimental path-space estimator will use the Donsker–Varadhan variational inequality:

$$
D_{KL}(P\|Q)
\ge
E_P[T]
-
\log E_Q[e^T],
$$

where

```text
P = forward-sequence distribution
Q = reversed-sequence distribution
T = learned critic
```

The critic will be trained only on training trajectories.

The Donsker–Varadhan objective used for reporting will be evaluated on held-out trajectory groups.

Training and evaluation data must therefore remain separated.

---

## 10. Fixed Critic

The critic architecture is fixed before outcome inspection as:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False,
)
    ->
StandardScaler()
    ->
LogisticRegression(
    max_iter=3000,
)
```

The quadratic feature expansion is included because irreversible path information may depend on interactions between coordinates and successive displacements.

A strictly linear critic could miss rotational or probability-current-like structure.

Degree two remains compact and interpretable.

There will be:

* no hyperparameter optimization;
* no neural network;
* no classifier zoo;
* no post-hoc model selection based on the observed MSC01 result.

---

## 11. Numerical Stability of the DV Objective

The term

$$
\log E_Q[e^T]
$$

must be evaluated numerically using a stable log-mean-exp calculation.

A naive calculation such as

```python
np.log(np.exp(scores).mean())
```

must not be relied upon when critic scores could produce numerical overflow or underflow.

The implementation must preserve finite numerical behavior for valid inputs.

---

## 12. Experimental Cross-Validation

Experimental critic fitting and evaluation will use:

```text
3-fold GroupKFold
```

The grouping variable is trajectory identity.

Forward and reversed versions of sequences derived from the same biological trajectory must remain in the same fold.

Overlapping sequence windows from the same track must not be randomly split across training and test data.

There must be zero trajectory-group overlap between training and held-out sets.

---

## 13. Experimental Lower-Bound Rate

For a sequence duration \(\Delta t\), the reported nonnegative summary is

$$
\dot I_{\mathrm{LB}}
=
\frac{
\max(0,\widehat D_{\mathrm{DV}})
}{
\Delta t
},
$$

where \(\widehat D_{\mathrm{DV}}\) is the held-out Donsker–Varadhan estimate.

The primary experimental name is:

> **coarse-grained path-space irreversibility lower-bound rate**

The units are:

```text
nats / min
```

The relevant sequence durations are:

```text
3-step sequence = 60 min
4-step sequence = 80 min
```

The unclipped held-out DV estimate must also be retained internally.

If it is negative because of finite-sample variability, that negative estimate must not be hidden or silently rewritten as evidence for a positive signal.

Clipping at zero is used only for the named nonnegative rate summary because the true KL divergence is nonnegative.

---

## 14. Primary Experimental Condition

The primary MSC01 condition is fixed before outcome inspection as:

```text
tracking source: CTC reference
coordinate representation: co-moving
sequence length: 3 steps
sequence duration: 60 min
```

This condition must be evaluated without changing the estimator because of its result.

---

## 15. Pre-Specified Experimental Sensitivity Conditions

The experimental analysis will evaluate exactly the following factors:

### Sequence length

```text
3-step
4-step
```

### Tracking source

```text
CTC reference
TrackMate
```

### Coordinate representation

```text
raw
co-moving
```

This gives:

$$
2 \times 2 \times 2 = 8
$$

pre-specified experimental conditions.

No fifth-step sequence analysis will be added.

No new biological dataset will be added during Phase 9.

No additional experimental condition will be introduced solely because the primary result is weak or negative.

---

## 16. Experimental Sequence Construction

Experimental three-step and four-step paths must reuse the existing tested sequence infrastructure in:

```text
src/sequences.py
```

including the current implementations of:

```text
build_k_step_sequences
get_k_step_sequence_array
reverse_k_step_sequences
```

Sequences must require the exact physical frame sequence

```text
t, t+1, ..., t+n
```

for the same cell.

Missing intermediate frames invalidate a sequence.

The time-reversed displacement sequence must preserve the existing project definition:

$$
(d_1,d_2,\ldots,d_n)
\rightarrow
(-d_n,\ldots,-d_2,-d_1).
$$

The Phase 9 analysis must not introduce row-offset approximations or a different reversal convention.

---

## 17. Raw and Co-Moving Representations

Raw and co-moving trajectories must remain separate experimental representations.

The co-moving transformation must preserve the historical project definition based on frame-wise population common motion.

If Phase 9 requires reconstruction of co-moving trajectories, the exact existing implementation must first be inspected and reused or promoted into tested source code.

The co-moving representation must not be described as the uniquely correct or true biological motion.

The population common-motion component must not be automatically described as microscope drift.

---

## 18. Null Calibration

Experimental null calibration will use paired orientation randomization.

For each exact forward/reverse pair:

```text
swap orientation with probability 0.5
```

The pair itself and its trajectory-group identity are preserved.

Cross-validation grouping must also be preserved.

The number of null replicates is fixed as:

```text
200
```

The null random seed is fixed as:

```text
2032
```

The finite-simulation empirical upper-tail p-value will use the established project convention:

$$
p=
\frac{
1+\#\{\text{null}\ge\text{observed}\}
}{
N_{\text{null}}+1
}.
$$

With 200 null replicates, the minimum attainable empirical p-value is:

$$
\frac{1}{201}
\approx
0.004975.
$$

An observed experimental lower-bound rate will be considered above its paired null only when it exceeds the null 95th percentile.

---

## 19. Interpretation Rules for MSC01

Possible contributors to observable forward/reverse trajectory asymmetry include:

* biological nonequilibrium dynamics;
* directional migration;
* population common motion;
* image-related effects;
* tracking reconstruction;
* coarse-graining;
* finite sample size.

CTC and TrackMate are alternative tracking reconstructions of the same microscopy sequence.

Agreement between them represents:

> tracking-source robustness

and must not be described as independent biological replication.

Absolute estimator differences between CTC and TrackMate must not automatically be interpreted as biological differences.

---

## 20. Negative-Result Policy

A null or weak MSC01 result is scientifically acceptable.

A null experimental result must not be interpreted as evidence for:

```text
thermodynamic equilibrium
detailed balance of the complete cell
microscopic reversibility
zero cellular entropy production
```

The synthetic coarse-graining analysis is specifically intended to demonstrate why absence of detectable irreversibility in a restricted observation space does not imply absence of microscopic dissipation.

The estimator must not be modified, replaced, or expanded solely to obtain a positive MSC01 result.

---

## 21. Explicitly Excluded Methods

The core Phase 9 analysis will not include:

* neural networks;
* random forests;
* gradient boosting;
* a classifier zoo;
* broad model comparison;
* hyperparameter optimization;
* five-step experimental paths;
* large state-discretization searches;
* post-hoc estimator selection;
* direct reinterpretation of Phase 5 displacement KL values as physical entropy-production rates.

---

## 22. Thermodynamic Uncertainty Relation

The Thermodynamic Uncertainty Relation is not part of the Phase 9 core analysis.

The current MSC01 trajectory analysis does not yet define a sufficiently well-justified integrated thermodynamic current, together with the necessary physical assumptions, to make a TUR-based entropy-production result clean and defensible.

TUR will therefore not be added merely because it appeared in an older project roadmap.

---

## 23. Planned Phase 9 Software

Phase 9 is expected to add:

```text
src/thermo.py
tests/test_thermo.py
notebooks/09_nonequilibrium_thermodynamics_MSC01.ipynb
```

Reusable mathematical and scientific definitions belong in:

```text
src/thermo.py
```

Scientific invariants and regression tests belong in:

```text
tests/test_thermo.py
```

The notebook is responsible for the auditable Phase 9 scientific workflow and stored outputs.

---

## 24. Testing Principles

Tests must target mathematical and scientific invariants rather than desired biological outcomes.

Planned test categories include:

1. expected symmetric and antisymmetric structure of the OU drift matrix;
2. analytical entropy production:

   * \(\omega=0 \rightarrow \sigma=0\);
   * changing the sign of \(\omega\) leaves \(\sigma\) unchanged;
3. reversing a path twice reproduces the original path;
4. valid Gaussian transition log densities are finite;
5. deliberately symmetric toy dynamics produce zero or near-zero path log-ratio;
6. stable log-mean-exp agrees with direct evaluation for numerically safe toy values;
7. a controlled finite discrete example respects the exact KL / DV lower-bound relationship within numerical tolerance;
8. grouped train/test splits have zero trajectory overlap;
9. forward/reverse orientation randomization preserves shapes and trajectory identities;
10. Phase 9 summary outputs contain no unexpected NaN or infinite values.

Tests must not encode the assumption that MSC01 has a positive irreversibility result.

---

## 25. Planned Core Figures

Phase 9 should remain visually compact.

Approximately three core figures are planned:

1. equilibrium versus nonequilibrium synthetic trajectories and/or probability-current behavior;
2. known synthetic entropy production together with the effect of coarse-graining;
3. experimental coarse-grained path-space irreversibility lower-bound rates across the eight pre-specified MSC01 conditions.

Additional figures should be created only when necessary for scientific validation or debugging, not for figure proliferation.

---

## 26. Phase 9 Stopping Rule

Phase 9 ends after completion of:

* the equilibrium synthetic benchmark;
* the nonequilibrium synthetic benchmark;
* analytical entropy-production validation;
* forward/reverse path-probability-ratio validation;
* the synthetic coarse-graining demonstration;
* validation of the variational KL lower bound on synthetic data;
* the primary CTC co-moving three-step experimental estimate;
* the remaining seven pre-specified experimental sensitivity conditions;
* paired orientation-null calibration;
* integrated scientific interpretation;
* Phase 9 automated tests;
* fresh-kernel execution of the complete Phase 9 notebook;
* Phase 9 standalone documentation;
* final Phase 9 Git merge and freeze.

No new estimator will be added solely because an experimental result is negative.

---

## 27. Final Reporting Boundary

At Phase 9 completion, results must be separated into three categories.

### Synthetic physical thermodynamics

For the explicitly defined Ornstein–Uhlenbeck model, physical entropy production can be calculated and named as such because the stochastic dynamics and thermodynamic interpretation are controlled.

### Experimental path-space analysis

For MSC01, the reported quantity is a:

> **coarse-grained path-space irreversibility lower-bound rate**

derived from observed centroid trajectories.

### Biological interpretation

The experimental result may inform whether the observed trajectory representation contains detectable time-direction information.

It must not be equated with the total physical entropy production of the living cell.

---

## 28. Analysis-Locking Statement

This document is committed before inspecting any Phase 9 scientific estimator output.

The Phase 9 experimental conditions, critic family, cross-validation structure, null design, null-replicate count, seeds, primary condition, interpretation boundaries, excluded methods, and stopping rule are therefore fixed in advance.

Subsequent changes are permitted only for genuine implementation errors, mathematical errors, or reproducibility problems.

Any such change must be explicitly documented and must not be motivated by whether an observed scientific result is positive or negative.
