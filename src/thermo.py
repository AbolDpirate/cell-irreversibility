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

