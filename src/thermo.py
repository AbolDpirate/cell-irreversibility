"""Core stochastic-thermodynamics utilities for Phase 9."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd

from sklearn.model_selection import GroupKFold

from src.classification import build_classification_dataset


def rotation_generator() -> np.ndarray:
    """Return the 2D counterclockwise rotation generator.

    R acts on a vector (x, y) as:

        R @ [x, y] = [-y, x]

    and satisfies:

        R.T = -R
        R.T @ R = I
    """
    return np.array(
        [
            [0.0, -1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )


def ou_drift_matrix(k: float, omega: float) -> np.ndarray:
    """Construct the drift matrix for the rotational 2D OU process.

    The model is

        dX = A X dt + sqrt(2D) dW

    with

        A = -k I + omega R.

    Parameters
    ----------
    k
        Positive restoring-rate parameter.
    omega
        Rotational driving rate. Its sign determines rotation direction.

    Returns
    -------
    numpy.ndarray
        A 2x2 drift matrix.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    identity = np.eye(2, dtype=float)
    rotation = rotation_generator()

    return -k * identity + omega * rotation


def stationary_covariance_isotropic(
    k: float,
    diffusion: float,
) -> np.ndarray:
    """Return the stationary covariance of the isotropic rotational OU model.

    For scalar diffusion coefficient D,

        C = (D / k) I.

    The rotational part does not change this stationary covariance.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if diffusion <= 0:
        raise ValueError("diffusion must be positive.")

    return (diffusion / k) * np.eye(2, dtype=float)


def analytic_epr_rotational_ou(
    k: float,
    omega: float,
) -> float:
    """Return the steady-state entropy-production rate of the model.

    For the isotropic 2D rotational Ornstein-Uhlenbeck process,

        sigma = 2 * omega**2 / k.

    The result is expressed in units of k_B per simulation-time unit.

    The scalar diffusion coefficient cancels from this analytical result.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    return float(2.0 * omega**2 / k)

def ou_transition_matrix(
    k: float,
    omega: float,
    dt: float,
) -> np.ndarray:
    """Return the exact finite-time transition matrix of the 2D OU process.

    For

        A = -k I + omega R,

    the exact transition over time interval dt is

        F = exp(A dt)
          = exp(-k dt) Rot(omega dt).

    Parameters
    ----------
    k
        Positive restoring-rate parameter.
    omega
        Rotational driving rate.
    dt
        Positive time interval.

    Returns
    -------
    numpy.ndarray
        A 2x2 exact transition matrix.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    decay = np.exp(-k * dt)
    angle = omega * dt

    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=float,
    )

    return decay * rotation


def ou_transition_covariance(
    k: float,
    diffusion: float,
    dt: float,
) -> np.ndarray:
    """Return the exact finite-time transition-noise covariance.

    For

        dX = A X dt + sqrt(2D) dW,

    with the isotropic rotational drift matrix,

        Q(dt)
        = (D / k) * (1 - exp(-2 k dt)) * I.

    Parameters
    ----------
    k
        Positive restoring-rate parameter.
    diffusion
        Positive scalar diffusion coefficient D.
    dt
        Positive time interval.

    Returns
    -------
    numpy.ndarray
        A 2x2 transition covariance matrix.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if diffusion <= 0:
        raise ValueError("diffusion must be positive.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    variance = (
        diffusion
        / k
        * (1.0 - np.exp(-2.0 * k * dt))
    )

    return variance * np.eye(2, dtype=float)

def simulate_rotational_ou(
    n_steps: int,
    k: float,
    omega: float,
    diffusion: float,
    dt: float,
    seed: int | None = 2031,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate the rotational OU process using its exact transition law.

    The process is

        dX = A X dt + sqrt(2D) dW,

    with

        A = -k I + omega R.

    The exact discrete-time transition is

        X_{n+1} = F X_n + eta_n,

    where

        F = exp(A dt)

    and

        eta_n ~ N(0, Q).

    If x0 is None, the initial state is sampled from the exact stationary
    Gaussian distribution.

    Parameters
    ----------
    n_steps
        Number of transition steps. The returned path therefore contains
        n_steps + 1 states.
    k
        Positive restoring-rate parameter.
    omega
        Rotational driving rate.
    diffusion
        Positive scalar diffusion coefficient D.
    dt
        Positive sampling interval.
    seed
        Seed for NumPy's local random-number generator.
    x0
        Optional initial 2D state. If omitted, the initial state is sampled
        from the stationary distribution.

    Returns
    -------
    numpy.ndarray
        Array with shape (n_steps + 1, 2).
    """
    if (
        isinstance(n_steps, bool)
        or not isinstance(n_steps, (int, np.integer))
        or n_steps < 1
    ):
        raise ValueError("n_steps must be a positive integer.")

    transition = ou_transition_matrix(
        k=k,
        omega=omega,
        dt=dt,
    )

    transition_covariance = ou_transition_covariance(
        k=k,
        diffusion=diffusion,
        dt=dt,
    )

    rng = np.random.default_rng(seed)

    path = np.empty(
        (n_steps + 1, 2),
        dtype=float,
    )

    if x0 is None:
        stationary_covariance = stationary_covariance_isotropic(
            k=k,
            diffusion=diffusion,
        )

        path[0] = rng.multivariate_normal(
            mean=np.zeros(2),
            cov=stationary_covariance,
        )

    else:
        x0_array = np.asarray(
            x0,
            dtype=float,
        )

        if x0_array.shape != (2,):
            raise ValueError("x0 must have shape (2,).")

        if not np.all(np.isfinite(x0_array)):
            raise ValueError("x0 must contain only finite values.")

        path[0] = x0_array

    noise = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=transition_covariance,
        size=n_steps,
    )

    for step_index in range(n_steps):
        path[step_index + 1] = (
            transition @ path[step_index]
            + noise[step_index]
        )

    return path

def rotational_increments(
    path: np.ndarray,
) -> np.ndarray:
    """Return signed rotational increments between successive 2D states.

    For successive states

        X_t     = (x_t, y_t)
        X_{t+1} = (x_{t+1}, y_{t+1}),

    the signed rotational increment is

        c_t = x_t * y_{t+1} - y_t * x_{t+1}.

    Positive values correspond to counterclockwise orientation.
    Negative values correspond to clockwise orientation.

    This quantity is a probability-current observable, not an
    entropy-production estimator.
    """
    path_array = np.asarray(
        path,
        dtype=float,
    )

    if (
        path_array.ndim != 2
        or path_array.shape[1] != 2
        or path_array.shape[0] < 2
    ):
        raise ValueError(
            "path must have shape (n_states, 2) with at least two states."
        )

    if not np.all(np.isfinite(path_array)):
        raise ValueError("path must contain only finite values.")

    current = path_array[:-1]
    following = path_array[1:]

    return (
        current[:, 0] * following[:, 1]
        - current[:, 1] * following[:, 0]
    )


def analytic_mean_rotational_increment(
    k: float,
    omega: float,
    diffusion: float,
    dt: float,
) -> float:
    """Return the stationary mean signed rotational increment.

    For the exactly sampled isotropic rotational OU process,

        E[c_t]
        = (2D / k)
          * exp(-k dt)
          * sin(omega dt).

    This is a discrete-time rotational-current observable and must not
    be interpreted as the physical entropy-production rate.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if diffusion <= 0:
        raise ValueError("diffusion must be positive.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    return float(
        2.0
        * diffusion
        / k
        * np.exp(-k * dt)
        * np.sin(omega * dt)
    )


def stationary_log_density_isotropic(
    state: np.ndarray,
    k: float,
    diffusion: float,
) -> float:
    """Return the stationary log density of a 2D isotropic OU state.

    The stationary covariance is

        C = (D / k) I.

    Therefore the stationary distribution is

        N(0, C).
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if diffusion <= 0:
        raise ValueError("diffusion must be positive.")

    state_array = np.asarray(
        state,
        dtype=float,
    )

    if state_array.shape != (2,):
        raise ValueError("state must have shape (2,).")

    if not np.all(np.isfinite(state_array)):
        raise ValueError("state must contain only finite values.")

    variance = diffusion / k

    quadratic = (
        state_array @ state_array
        / variance
    )

    return float(
        -np.log(2.0 * np.pi * variance)
        - 0.5 * quadratic
    )


def ou_transition_log_density(
    current_state: np.ndarray,
    next_state: np.ndarray,
    k: float,
    omega: float,
    diffusion: float,
    dt: float,
) -> float:
    """Return the exact OU transition log density.

    Computes

        log p(X_{t+dt} = next_state | X_t = current_state)

    using the exact Gaussian finite-time transition law.
    """
    current = np.asarray(
        current_state,
        dtype=float,
    )

    following = np.asarray(
        next_state,
        dtype=float,
    )

    if current.shape != (2,):
        raise ValueError(
            "current_state must have shape (2,)."
        )

    if following.shape != (2,):
        raise ValueError(
            "next_state must have shape (2,)."
        )

    if (
        not np.all(np.isfinite(current))
        or not np.all(np.isfinite(following))
    ):
        raise ValueError(
            "states must contain only finite values."
        )

    transition = ou_transition_matrix(
        k=k,
        omega=omega,
        dt=dt,
    )

    transition_covariance = ou_transition_covariance(
        k=k,
        diffusion=diffusion,
        dt=dt,
    )

    variance = transition_covariance[0, 0]

    residual = (
        following
        - transition @ current
    )

    quadratic = (
        residual @ residual
        / variance
    )

    return float(
        -np.log(2.0 * np.pi * variance)
        - 0.5 * quadratic
    )


def analytic_sampled_path_irreversibility_rate(
    k: float,
    omega: float,
    dt: float,
) -> float:
    """Return the exact irreversibility rate of the sampled OU chain.

    For observations separated by dt, the stationary forward/reverse
    path-space KL rate is

        4 exp(-2 k dt) sin^2(omega dt)
        --------------------------------
        (1 - exp(-2 k dt)) dt

    in nats per simulation-time unit.

    As dt -> 0, this approaches the continuous-time physical EPR

        2 omega^2 / k.

    At finite dt this is a sampled path-space irreversibility rate,
    not the continuous-time physical entropy-production rate.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    decay_squared = np.exp(
        -2.0 * k * dt
    )

    one_minus_decay_squared = (
        -np.expm1(-2.0 * k * dt)
    )

    numerator = (
        4.0
        * decay_squared
        * np.sin(omega * dt) ** 2
    )

    denominator = (
        one_minus_decay_squared
        * dt
    )

    return float(
        numerator / denominator
    )

def reverse_state_path(
    path: np.ndarray,
) -> np.ndarray:
    """Return the time-reversed order of a 2D state path.

    For a position path

        (x_0, x_1, ..., x_N),

    the reversed path is

        (x_N, x_{N-1}, ..., x_0).

    Position is even under time reversal, so the state values themselves
    are not sign-flipped; only their temporal order is reversed.
    """
    path_array = np.asarray(
        path,
        dtype=float,
    )

    if (
        path_array.ndim != 2
        or path_array.shape[1] != 2
        or path_array.shape[0] < 2
    ):
        raise ValueError(
            "path must have shape (n_states, 2) with at least two states."
        )

    if not np.all(np.isfinite(path_array)):
        raise ValueError(
            "path must contain only finite values."
        )

    return path_array[::-1].copy()


def ou_path_log_probability(
    path: np.ndarray,
    k: float,
    omega: float,
    diffusion: float,
    dt: float,
) -> float:
    """Return the stationary log probability of a complete OU path.

    The path probability is

        P[path]
        = pi(x_0)
          * product_t p(x_{t+1} | x_t),

    where pi is the stationary state density and the transition density
    is the exact finite-time Gaussian OU transition.
    """
    path_array = np.asarray(
        path,
        dtype=float,
    )

    if (
        path_array.ndim != 2
        or path_array.shape[1] != 2
        or path_array.shape[0] < 2
    ):
        raise ValueError(
            "path must have shape (n_states, 2) with at least two states."
        )

    if not np.all(np.isfinite(path_array)):
        raise ValueError(
            "path must contain only finite values."
        )

    transition = ou_transition_matrix(
        k=k,
        omega=omega,
        dt=dt,
    )

    transition_covariance = ou_transition_covariance(
        k=k,
        diffusion=diffusion,
        dt=dt,
    )

    variance = transition_covariance[0, 0]

    current_states = path_array[:-1]
    next_states = path_array[1:]

    conditional_means = (
        current_states @ transition.T
    )

    residuals = (
        next_states - conditional_means
    )

    squared_residual_norms = np.sum(
        residuals**2,
        axis=1,
    )

    transition_log_probabilities = (
        -np.log(2.0 * np.pi * variance)
        - 0.5
        * squared_residual_norms
        / variance
    )

    initial_log_probability = (
        stationary_log_density_isotropic(
            state=path_array[0],
            k=k,
            diffusion=diffusion,
        )
    )

    return float(
        initial_log_probability
        + transition_log_probabilities.sum()
    )


def ou_path_log_ratio(
    path: np.ndarray,
    k: float,
    omega: float,
    diffusion: float,
    dt: float,
) -> float:
    """Return log P[path] - log P[reversed path].

    This quantity is the forward-versus-reversed path log-probability
    ratio for the exactly sampled stationary OU process.
    """
    forward_log_probability = (
        ou_path_log_probability(
            path=path,
            k=k,
            omega=omega,
            diffusion=diffusion,
            dt=dt,
        )
    )

    reversed_path = reverse_state_path(path)

    reversed_log_probability = (
        ou_path_log_probability(
            path=reversed_path,
            k=k,
            omega=omega,
            diffusion=diffusion,
            dt=dt,
        )
    )

    return float(
        forward_log_probability
        - reversed_log_probability
    )

def projected_scalar_covariance_matrix(
    n_states: int,
    k: float,
    omega: float,
    diffusion: float,
    dt: float,
) -> np.ndarray:
    """Return the exact covariance matrix of one observed Cartesian coordinate.

    For either x or y of the stationary isotropic rotational OU process,

        Cov[X_t, X_{t+m}]
        = (D / k)
          * exp(-k * |m| * dt)
          * cos(omega * |m| * dt).

    The one-coordinate process is generally not first-order Markov,
    so this full covariance matrix is used instead of a one-step
    transition approximation.
    """
    if (
        isinstance(n_states, bool)
        or not isinstance(n_states, (int, np.integer))
        or n_states < 2
    ):
        raise ValueError(
            "n_states must be an integer of at least 2."
        )

    if k <= 0:
        raise ValueError("k must be positive.")

    if diffusion <= 0:
        raise ValueError("diffusion must be positive.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    indices = np.arange(n_states)

    lag_matrix = np.abs(
        indices[:, None] - indices[None, :]
    )

    lag_times = lag_matrix * dt

    stationary_variance = diffusion / k

    return (
        stationary_variance
        * np.exp(-k * lag_times)
        * np.cos(omega * lag_times)
    )


def zero_mean_gaussian_log_density(
    values: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return the log density of a zero-mean multivariate Gaussian."""
    values_array = np.asarray(
        values,
        dtype=float,
    )

    covariance_array = np.asarray(
        covariance,
        dtype=float,
    )

    if values_array.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")

    if covariance_array.shape != (
        values_array.size,
        values_array.size,
    ):
        raise ValueError(
            "covariance shape must match the number of values."
        )

    if (
        not np.all(np.isfinite(values_array))
        or not np.all(np.isfinite(covariance_array))
    ):
        raise ValueError(
            "values and covariance must contain only finite values."
        )

    sign, log_determinant = np.linalg.slogdet(
        covariance_array
    )

    if sign <= 0:
        raise ValueError(
            "covariance must be positive definite."
        )

    solved = np.linalg.solve(
        covariance_array,
        values_array,
    )

    quadratic = values_array @ solved

    dimension = values_array.size

    return float(
        -0.5
        * (
            dimension * np.log(2.0 * np.pi)
            + log_determinant
            + quadratic
        )
    )


def projected_scalar_path_log_ratio(
    values: np.ndarray,
    k: float,
    omega: float,
    diffusion: float,
    dt: float,
) -> float:
    """Return the exact forward/reverse log ratio after hiding one coordinate.

    The observed scalar path is evaluated using its full stationary
    multivariate-Gaussian distribution, not a false first-order
    Markov approximation.
    """
    values_array = np.asarray(
        values,
        dtype=float,
    )

    if values_array.ndim != 1 or values_array.size < 2:
        raise ValueError(
            "values must be a one-dimensional path with at least two states."
        )

    if not np.all(np.isfinite(values_array)):
        raise ValueError(
            "values must contain only finite values."
        )

    covariance = projected_scalar_covariance_matrix(
        n_states=values_array.size,
        k=k,
        omega=omega,
        diffusion=diffusion,
        dt=dt,
    )

    forward = zero_mean_gaussian_log_density(
        values=values_array,
        covariance=covariance,
    )

    reverse = zero_mean_gaussian_log_density(
        values=values_array[::-1],
        covariance=covariance,
    )

    return float(forward - reverse)

def stable_log_mean_exp(
    scores: np.ndarray,
) -> float:
    """Compute log(mean(exp(scores))) in a numerically stable way."""
    scores_array = np.asarray(
        scores,
        dtype=float,
    )

    if scores_array.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional."
        )

    if scores_array.size == 0:
        raise ValueError(
            "scores must not be empty."
        )

    if not np.all(np.isfinite(scores_array)):
        raise ValueError(
            "scores must contain only finite values."
        )

    maximum = np.max(scores_array)

    return float(
        maximum
        + np.log(
            np.mean(
                np.exp(
                    scores_array - maximum
                )
            )
        )
    )


def donsker_varadhan_lower_bound(
    forward_scores: np.ndarray,
    reverse_scores: np.ndarray,
) -> float:
    """Evaluate the empirical Donsker-Varadhan KL lower bound.

    For critic T,

        D_KL(P || Q)
        >= E_P[T] - log(E_Q[exp(T)]).

    Here forward_scores are critic values evaluated on samples from P,
    and reverse_scores are critic values evaluated on samples from Q.
    """
    forward_array = np.asarray(
        forward_scores,
        dtype=float,
    )

    reverse_array = np.asarray(
        reverse_scores,
        dtype=float,
    )

    if (
        forward_array.ndim != 1
        or reverse_array.ndim != 1
    ):
        raise ValueError(
            "forward_scores and reverse_scores must be one-dimensional."
        )

    if (
        forward_array.size == 0
        or reverse_array.size == 0
    ):
        raise ValueError(
            "forward_scores and reverse_scores must not be empty."
        )

    if (
        not np.all(np.isfinite(forward_array))
        or not np.all(np.isfinite(reverse_array))
    ):
        raise ValueError(
            "critic scores must contain only finite values."
        )

    return float(
        np.mean(forward_array)
        - stable_log_mean_exp(reverse_array)
    )


def make_quadratic_logistic_critic() -> Pipeline:
    """Construct the fixed Phase 9 variational critic.

    The architecture is pre-specified as:

        PolynomialFeatures(degree=2, include_bias=False)
            ->
        StandardScaler()
            ->
        LogisticRegression(max_iter=3000)

    No hyperparameter optimization is performed.
    """
    return Pipeline(
        steps=[
            (
                "polynomial",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=3000,
                ),
            ),
        ]
    )

def evaluate_grouped_dv_critic(
    forward_array: np.ndarray,
    reverse_array: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Evaluate the fixed Phase 9 DV critic on held-out trajectory groups.

    Forward and reversed examples are combined into a balanced binary
    classification dataset.

    GroupKFold is applied using trajectory identity, so all examples
    from a trajectory remain entirely in either the training or the
    held-out fold.

    For each fold:

    1. fit the fixed quadratic-logistic critic on training groups;
    2. obtain critic scores on held-out examples only;
    3. separate held-out forward and reverse scores;
    4. evaluate the held-out Donsker-Varadhan estimate.

    Returns both the raw finite-sample DV estimate and its nonnegative
    clipped summary.
    """
    forward = np.asarray(
        forward_array,
        dtype=float,
    )

    reverse = np.asarray(
        reverse_array,
        dtype=float,
    )

    groups_array = np.asarray(groups)

    if forward.ndim != 2:
        raise ValueError(
            "forward_array must be two-dimensional."
        )

    if reverse.ndim != 2:
        raise ValueError(
            "reverse_array must be two-dimensional."
        )

    if forward.shape != reverse.shape:
        raise ValueError(
            "forward_array and reverse_array must have identical shapes."
        )

    if forward.shape[0] == 0:
        raise ValueError(
            "forward_array and reverse_array must not be empty."
        )

    if forward.shape[1] == 0:
        raise ValueError(
            "arrays must contain at least one feature."
        )

    if (
        not np.all(np.isfinite(forward))
        or not np.all(np.isfinite(reverse))
    ):
        raise ValueError(
            "forward and reverse arrays must contain only finite values."
        )

    if groups_array.ndim != 1:
        raise ValueError(
            "groups must be one-dimensional."
        )

    if groups_array.shape[0] != forward.shape[0]:
        raise ValueError(
            "groups must contain one entry per forward/reverse pair."
        )

    if pd.isna(groups_array).any():
        raise ValueError(
            "groups must not contain missing values."
        )

    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, (int, np.integer))
        or n_splits < 2
    ):
        raise ValueError(
            "n_splits must be an integer of at least 2."
        )

    unique_groups = np.unique(groups_array)

    if unique_groups.size < n_splits:
        raise ValueError(
            "number of unique groups must be at least n_splits."
        )

    X, y, doubled_groups = build_classification_dataset(
        forward_array=forward,
        reverse_array=reverse,
        cell_ids=groups_array,
    )

    cv = GroupKFold(
        n_splits=n_splits,
    )

    fold_rows = []

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups=doubled_groups,
        ),
        start=1,
    ):
        train_groups = np.unique(
            doubled_groups[train_indices]
        )

        test_groups = np.unique(
            doubled_groups[test_indices]
        )

        overlap = np.intersect1d(
            train_groups,
            test_groups,
        )

        if overlap.size != 0:
            raise RuntimeError(
                "Group leakage detected between training and held-out data."
            )

        critic = make_quadratic_logistic_critic()

        critic.fit(
            X[train_indices],
            y[train_indices],
        )

        held_out_scores = critic.decision_function(
            X[test_indices]
        )

        held_out_labels = y[test_indices]

        forward_scores = held_out_scores[
            held_out_labels == 1
        ]

        reverse_scores = held_out_scores[
            held_out_labels == 0
        ]

        dv_raw = donsker_varadhan_lower_bound(
            forward_scores=forward_scores,
            reverse_scores=reverse_scores,
        )

        fold_rows.append(
            {
                "fold": fold_number,
                "n_train_examples": train_indices.size,
                "n_test_examples": test_indices.size,
                "n_train_groups": train_groups.size,
                "n_test_groups": test_groups.size,
                "n_test_forward": forward_scores.size,
                "n_test_reverse": reverse_scores.size,
                "group_overlap": overlap.size,
                "dv_raw": dv_raw,
                "dv_clipped": max(
                    0.0,
                    dv_raw,
                ),
            }
        )

    return pd.DataFrame(
        fold_rows
    )

