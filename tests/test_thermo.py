import numpy as np
import pytest

from src.thermo import (
    analytic_epr_rotational_ou,
    ou_drift_matrix,
    rotation_generator,
    stationary_covariance_isotropic,
)


def test_rotation_generator_is_antisymmetric():
    rotation = rotation_generator()

    np.testing.assert_allclose(
        rotation.T,
        -rotation,
    )


def test_rotation_generator_has_unit_metric():
    rotation = rotation_generator()

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(2),
    )


def test_ou_drift_matrix_has_expected_symmetric_and_antisymmetric_parts():
    k = 1.5
    omega = 0.7

    drift = ou_drift_matrix(
        k=k,
        omega=omega,
    )

    symmetric = 0.5 * (drift + drift.T)
    antisymmetric = 0.5 * (drift - drift.T)

    np.testing.assert_allclose(
        symmetric,
        -k * np.eye(2),
    )

    np.testing.assert_allclose(
        antisymmetric,
        omega * rotation_generator(),
    )


def test_stationary_covariance_matches_isotropic_solution():
    covariance = stationary_covariance_isotropic(
        k=2.0,
        diffusion=3.0,
    )

    expected = 1.5 * np.eye(2)

    np.testing.assert_allclose(
        covariance,
        expected,
    )


def test_equilibrium_entropy_production_is_zero():
    sigma = analytic_epr_rotational_ou(
        k=1.0,
        omega=0.0,
    )

    assert sigma == pytest.approx(0.0)


def test_nonequilibrium_baseline_entropy_production_is_two():
    sigma = analytic_epr_rotational_ou(
        k=1.0,
        omega=1.0,
    )

    assert sigma == pytest.approx(2.0)


def test_entropy_production_is_independent_of_rotation_direction():
    sigma_positive = analytic_epr_rotational_ou(
        k=1.0,
        omega=1.25,
    )

    sigma_negative = analytic_epr_rotational_ou(
        k=1.0,
        omega=-1.25,
    )

    assert sigma_positive == pytest.approx(sigma_negative)


def test_invalid_restoring_rate_is_rejected():
    with pytest.raises(ValueError):
        ou_drift_matrix(
            k=0.0,
            omega=1.0,
        )

    with pytest.raises(ValueError):
        analytic_epr_rotational_ou(
            k=-1.0,
            omega=1.0,
        )


def test_invalid_diffusion_is_rejected():
    with pytest.raises(ValueError):
        stationary_covariance_isotropic(
            k=1.0,
            diffusion=0.0,
        )

