"""Core stochastic-thermodynamics utilities for Phase 9."""

from __future__ import annotations

import numpy as np


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
