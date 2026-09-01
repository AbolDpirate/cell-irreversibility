import numpy as np
import pytest

from src.thermo import (
    analytic_epr_rotational_ou,
    ou_drift_matrix,
    ou_transition_covariance,
    ou_transition_matrix,
    rotation_generator,
    simulate_rotational_ou,
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

def test_stationary_covariance_satisfies_continuous_lyapunov_equation():
    k = 1.3
    omega = 0.9
    diffusion = 0.8

    drift = ou_drift_matrix(
        k=k,
        omega=omega,
    )

    covariance = stationary_covariance_isotropic(
        k=k,
        diffusion=diffusion,
    )

    residual = (
        drift @ covariance
        + covariance @ drift.T
        + 2.0 * diffusion * np.eye(2)
    )

    np.testing.assert_allclose(
        residual,
        np.zeros((2, 2)),
        atol=1e-12,
    )


def test_exact_transition_matrix_matches_rotation_and_contraction():
    k = 0.7
    omega = 1.2
    dt = 0.4

    transition = ou_transition_matrix(
        k=k,
        omega=omega,
        dt=dt,
    )

    decay = np.exp(-k * dt)
    angle = omega * dt

    expected = decay * np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    np.testing.assert_allclose(
        transition,
        expected,
    )


def test_exact_transition_covariance_matches_closed_form():
    k = 1.4
    diffusion = 0.6
    dt = 0.25

    covariance = ou_transition_covariance(
        k=k,
        diffusion=diffusion,
        dt=dt,
    )

    expected_variance = (
        diffusion
        / k
        * (1.0 - np.exp(-2.0 * k * dt))
    )

    np.testing.assert_allclose(
        covariance,
        expected_variance * np.eye(2),
    )


def test_stationary_covariance_is_invariant_under_exact_transition():
    k = 1.1
    omega = 0.8
    diffusion = 0.9
    dt = 0.3

    stationary = stationary_covariance_isotropic(
        k=k,
        diffusion=diffusion,
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

    propagated = (
        transition
        @ stationary
        @ transition.T
        + transition_covariance
    )

    np.testing.assert_allclose(
        propagated,
        stationary,
        atol=1e-12,
    )


def test_invalid_transition_time_is_rejected():
    with pytest.raises(ValueError):
        ou_transition_matrix(
            k=1.0,
            omega=1.0,
            dt=0.0,
        )

    with pytest.raises(ValueError):
        ou_transition_covariance(
            k=1.0,
            diffusion=1.0,
            dt=-0.1,
        )

def test_simulated_ou_path_has_expected_shape_and_finite_values():
    path = simulate_rotational_ou(
        n_steps=20,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
        seed=2031,
    )

    assert path.shape == (21, 2)
    assert np.all(np.isfinite(path))


def test_simulation_is_reproducible_for_fixed_seed():
    path_a = simulate_rotational_ou(
        n_steps=25,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
        seed=2031,
    )

    path_b = simulate_rotational_ou(
        n_steps=25,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
        seed=2031,
    )

    np.testing.assert_allclose(
        path_a,
        path_b,
    )


def test_supplied_initial_state_is_preserved():
    x0 = np.array(
        [1.25, -0.75],
        dtype=float,
    )

    path = simulate_rotational_ou(
        n_steps=10,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
        seed=2031,
        x0=x0,
    )

    np.testing.assert_allclose(
        path[0],
        x0,
    )


def test_one_step_simulation_matches_exact_transition_law():
    k = 1.0
    omega = 0.8
    diffusion = 0.7
    dt = 0.2
    seed = 1234

    x0 = np.array(
        [0.4, -1.1],
        dtype=float,
    )

    path = simulate_rotational_ou(
        n_steps=1,
        k=k,
        omega=omega,
        diffusion=diffusion,
        dt=dt,
        seed=seed,
        x0=x0,
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

    rng = np.random.default_rng(seed)

    expected_noise = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=transition_covariance,
    )

    expected_next = (
        transition @ x0
        + expected_noise
    )

    np.testing.assert_allclose(
        path[1],
        expected_next,
    )


def test_invalid_simulation_arguments_are_rejected():
    with pytest.raises(ValueError):
        simulate_rotational_ou(
            n_steps=0,
            k=1.0,
            omega=1.0,
            diffusion=1.0,
            dt=0.1,
        )

    with pytest.raises(ValueError):
        simulate_rotational_ou(
            n_steps=10,
            k=1.0,
            omega=1.0,
            diffusion=1.0,
            dt=0.1,
            x0=np.array([1.0, 2.0, 3.0]),
        )

