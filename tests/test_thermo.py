import numpy as np
import pytest

from src.thermo import (
    analytic_epr_rotational_ou,
    analytic_mean_rotational_increment,
    analytic_sampled_path_irreversibility_rate,
    ou_drift_matrix,
    ou_path_log_probability,
    ou_path_log_ratio,
    ou_transition_covariance,
    ou_transition_log_density,
    ou_transition_matrix,
    reverse_state_path,
    rotation_generator,
    rotational_increments,
    simulate_rotational_ou,
    stationary_covariance_isotropic,
    stationary_log_density_isotropic,
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

def test_rotational_increments_have_expected_sign_for_simple_path():
    path = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    increments = rotational_increments(path)

    np.testing.assert_allclose(
        increments,
        np.array([1.0, 1.0]),
    )


def test_equilibrium_mean_rotational_increment_is_zero():
    expected = analytic_mean_rotational_increment(
        k=1.0,
        omega=0.0,
        diffusion=1.0,
        dt=0.1,
    )

    assert expected == pytest.approx(0.0)


def test_mean_rotational_increment_changes_sign_with_rotation_direction():
    positive = analytic_mean_rotational_increment(
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
    )

    negative = analytic_mean_rotational_increment(
        k=1.0,
        omega=-1.0,
        diffusion=1.0,
        dt=0.1,
    )

    assert positive > 0.0
    assert negative < 0.0
    assert positive == pytest.approx(-negative)


def test_invalid_rotational_increment_path_is_rejected():
    with pytest.raises(ValueError):
        rotational_increments(
            np.array([1.0, 2.0])
        )

    with pytest.raises(ValueError):
        rotational_increments(
            np.array([[1.0, np.nan], [2.0, 3.0]])
        )

def test_stationary_log_density_at_origin_matches_known_gaussian():
    log_density = stationary_log_density_isotropic(
        state=np.array([0.0, 0.0]),
        k=1.0,
        diffusion=1.0,
    )

    expected = -np.log(2.0 * np.pi)

    assert log_density == pytest.approx(expected)


def test_transition_log_density_is_maximal_at_exact_conditional_mean():
    k = 1.0
    omega = 0.8
    diffusion = 0.7
    dt = 0.2

    current = np.array(
        [0.4, -1.1],
        dtype=float,
    )

    transition = ou_transition_matrix(
        k=k,
        omega=omega,
        dt=dt,
    )

    conditional_mean = (
        transition @ current
    )

    transition_covariance = (
        ou_transition_covariance(
            k=k,
            diffusion=diffusion,
            dt=dt,
        )
    )

    variance = transition_covariance[0, 0]

    observed = ou_transition_log_density(
        current_state=current,
        next_state=conditional_mean,
        k=k,
        omega=omega,
        diffusion=diffusion,
        dt=dt,
    )

    expected = -np.log(
        2.0 * np.pi * variance
    )

    assert observed == pytest.approx(expected)


def test_sampled_irreversibility_rate_is_zero_at_equilibrium():
    rate = analytic_sampled_path_irreversibility_rate(
        k=1.0,
        omega=0.0,
        dt=0.1,
    )

    assert rate == pytest.approx(0.0)


def test_sampled_irreversibility_rate_is_independent_of_rotation_direction():
    positive = analytic_sampled_path_irreversibility_rate(
        k=1.0,
        omega=1.2,
        dt=0.1,
    )

    negative = analytic_sampled_path_irreversibility_rate(
        k=1.0,
        omega=-1.2,
        dt=0.1,
    )

    assert positive == pytest.approx(negative)


def test_sampled_irreversibility_rate_approaches_continuous_epr():
    k = 1.0
    omega = 1.0
    dt = 1e-6

    sampled = analytic_sampled_path_irreversibility_rate(
        k=k,
        omega=omega,
        dt=dt,
    )

    continuous = analytic_epr_rotational_ou(
        k=k,
        omega=omega,
    )

    assert sampled == pytest.approx(
        continuous,
        rel=1e-5,
    )


def test_finite_sampling_reduces_observable_irreversibility():
    k = 1.0
    omega = 1.0
    dt = 0.1

    sampled = analytic_sampled_path_irreversibility_rate(
        k=k,
        omega=omega,
        dt=dt,
    )

    continuous = analytic_epr_rotational_ou(
        k=k,
        omega=omega,
    )

    assert sampled > 0.0
    assert sampled < continuous

def test_reversing_state_path_twice_returns_original():
    path = np.array(
        [
            [0.2, -0.4],
            [1.0, 0.3],
            [-0.6, 0.8],
            [0.1, 1.2],
        ],
        dtype=float,
    )

    reversed_twice = reverse_state_path(
        reverse_state_path(path)
    )

    np.testing.assert_allclose(
        reversed_twice,
        path,
    )


def test_path_log_probability_matches_manual_transition_sum():
    path = np.array(
        [
            [0.3, -0.2],
            [0.8, 0.1],
            [0.4, 0.7],
        ],
        dtype=float,
    )

    k = 1.0
    omega = 0.6
    diffusion = 0.9
    dt = 0.2

    observed = ou_path_log_probability(
        path=path,
        k=k,
        omega=omega,
        diffusion=diffusion,
        dt=dt,
    )

    expected = stationary_log_density_isotropic(
        state=path[0],
        k=k,
        diffusion=diffusion,
    )

    expected += ou_transition_log_density(
        current_state=path[0],
        next_state=path[1],
        k=k,
        omega=omega,
        diffusion=diffusion,
        dt=dt,
    )

    expected += ou_transition_log_density(
        current_state=path[1],
        next_state=path[2],
        k=k,
        omega=omega,
        diffusion=diffusion,
        dt=dt,
    )

    assert observed == pytest.approx(expected)


def test_equilibrium_path_log_ratio_is_zero():
    path = np.array(
        [
            [0.2, -0.3],
            [1.1, 0.4],
            [-0.5, 0.7],
            [0.9, -0.2],
        ],
        dtype=float,
    )

    log_ratio = ou_path_log_ratio(
        path=path,
        k=1.0,
        omega=0.0,
        diffusion=1.0,
        dt=0.1,
    )

    assert log_ratio == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_path_log_ratio_changes_sign_under_path_reversal():
    path = np.array(
        [
            [0.2, -0.3],
            [1.1, 0.4],
            [-0.5, 0.7],
            [0.9, -0.2],
        ],
        dtype=float,
    )

    forward_ratio = ou_path_log_ratio(
        path=path,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
    )

    reverse_ratio = ou_path_log_ratio(
        path=reverse_state_path(path),
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
    )

    assert forward_ratio == pytest.approx(
        -reverse_ratio,
        abs=1e-12,
    )


def test_path_log_ratio_matches_rotational_increment_identity():
    path = np.array(
        [
            [0.2, -0.4],
            [0.9, 0.1],
            [0.5, 0.8],
            [-0.1, 0.6],
        ],
        dtype=float,
    )

    k = 1.1
    omega = 0.7
    diffusion = 0.8
    dt = 0.2

    observed = ou_path_log_ratio(
        path=path,
        k=k,
        omega=omega,
        diffusion=diffusion,
        dt=dt,
    )

    decay = np.exp(-k * dt)

    stationary_variance = (
        diffusion / k
    )

    one_minus_decay_squared = (
        -np.expm1(-2.0 * k * dt)
    )

    coefficient = (
        2.0
        * decay
        * np.sin(omega * dt)
        / (
            stationary_variance
            * one_minus_decay_squared
        )
    )

    expected = (
        coefficient
        * rotational_increments(path).sum()
    )

    assert observed == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_invalid_state_path_is_rejected():
    invalid_path = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    with pytest.raises(ValueError):
        reverse_state_path(invalid_path)

    with pytest.raises(ValueError):
        ou_path_log_probability(
            path=invalid_path,
            k=1.0,
            omega=1.0,
            diffusion=1.0,
            dt=0.1,
        )

