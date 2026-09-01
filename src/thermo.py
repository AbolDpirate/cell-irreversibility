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
