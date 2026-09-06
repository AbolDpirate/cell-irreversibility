import numpy as np
import pandas as pd
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
    projected_scalar_covariance_matrix,
    projected_scalar_path_log_ratio,
    zero_mean_gaussian_log_density,
    donsker_varadhan_lower_bound,
    make_quadratic_logistic_critic,
    stable_log_mean_exp,
    evaluate_grouped_dv_critic,
    build_grouped_ou_path_samples,
    summarize_grouped_dv,
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

def test_projected_scalar_covariance_matches_known_lags():
    covariance = projected_scalar_covariance_matrix(
        n_states=3,
        k=1.0,
        omega=0.7,
        diffusion=1.0,
        dt=0.2,
    )

    variance = 1.0

    lag_one = (
        np.exp(-0.2)
        * np.cos(0.7 * 0.2)
    )

    lag_two = (
        np.exp(-0.4)
        * np.cos(0.7 * 0.4)
    )

    expected = np.array(
        [
            [variance, lag_one, lag_two],
            [lag_one, variance, lag_one],
            [lag_two, lag_one, variance],
        ]
    )

    np.testing.assert_allclose(
        covariance,
        expected,
    )


def test_projected_scalar_covariance_is_invariant_under_time_reversal():
    covariance = projected_scalar_covariance_matrix(
        n_states=7,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
    )

    reversed_covariance = covariance[
        ::-1,
        ::-1,
    ]

    np.testing.assert_allclose(
        reversed_covariance,
        covariance,
        atol=1e-12,
    )


def test_projected_scalar_path_log_ratio_is_zero_even_when_full_system_is_nonequilibrium():
    values = np.array(
        [
            0.3,
            1.1,
            -0.2,
            0.8,
            -0.6,
            0.1,
        ],
        dtype=float,
    )

    log_ratio = projected_scalar_path_log_ratio(
        values=values,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
    )

    assert log_ratio == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_invalid_projected_scalar_inputs_are_rejected():
    with pytest.raises(ValueError):
        projected_scalar_covariance_matrix(
            n_states=1,
            k=1.0,
            omega=1.0,
            diffusion=1.0,
            dt=0.1,
        )

    with pytest.raises(ValueError):
        projected_scalar_path_log_ratio(
            values=np.array([1.0]),
            k=1.0,
            omega=1.0,
            diffusion=1.0,
            dt=0.1,
        )

def test_stable_log_mean_exp_matches_direct_calculation_on_safe_values():
    scores = np.array(
        [-1.0, 0.0, 1.0],
        dtype=float,
    )

    observed = stable_log_mean_exp(
        scores
    )

    expected = np.log(
        np.mean(
            np.exp(scores)
        )
    )

    assert observed == pytest.approx(
        expected
    )


def test_stable_log_mean_exp_remains_finite_for_large_scores():
    scores = np.array(
        [1000.0, 1001.0, 999.0],
        dtype=float,
    )

    observed = stable_log_mean_exp(
        scores
    )

    assert np.isfinite(observed)


def test_dv_lower_bound_recovers_exact_discrete_toy_kl():
    log_three = np.log(3.0)

    forward_scores = np.array(
        [
            log_three,
            log_three,
            log_three,
            -log_three,
        ],
        dtype=float,
    )

    reverse_scores = np.array(
        [
            log_three,
            -log_three,
            -log_three,
            -log_three,
        ],
        dtype=float,
    )

    observed = donsker_varadhan_lower_bound(
        forward_scores=forward_scores,
        reverse_scores=reverse_scores,
    )

    expected_kl = 0.5 * log_three

    assert observed == pytest.approx(
        expected_kl
    )


def test_dv_lower_bound_is_invariant_to_additive_critic_shift():
    forward_scores = np.array(
        [0.2, 0.8, 1.1, -0.3],
        dtype=float,
    )

    reverse_scores = np.array(
        [-0.4, 0.1, 0.3, -0.8],
        dtype=float,
    )

    original = donsker_varadhan_lower_bound(
        forward_scores=forward_scores,
        reverse_scores=reverse_scores,
    )

    shift = 17.5

    shifted = donsker_varadhan_lower_bound(
        forward_scores=(
            forward_scores + shift
        ),
        reverse_scores=(
            reverse_scores + shift
        ),
    )

    assert shifted == pytest.approx(
        original
    )


def test_fixed_quadratic_logistic_critic_matches_prespecification():
    critic = make_quadratic_logistic_critic()

    polynomial = critic.named_steps[
        "polynomial"
    ]

    logistic = critic.named_steps[
        "logistic"
    ]

    assert polynomial.degree == 2
    assert polynomial.include_bias is False
    assert logistic.max_iter == 3000


def test_invalid_dv_inputs_are_rejected():
    with pytest.raises(ValueError):
        stable_log_mean_exp(
            np.array([])
        )

    with pytest.raises(ValueError):
        donsker_varadhan_lower_bound(
            forward_scores=np.array(
                [1.0, np.nan]
            ),
            reverse_scores=np.array(
                [0.0, 1.0]
            ),
        )


def test_grouped_dv_evaluator_has_zero_group_leakage_and_balanced_test_labels():
    forward = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, -0.1],
            [1.2, 0.2],
            [0.8, -0.2],
            [1.05, 0.05],
        ],
        dtype=float,
    )

    reverse = -forward

    groups = np.array(
        [0, 1, 2, 3, 4, 5]
    )

    result = evaluate_grouped_dv_critic(
        forward_array=forward,
        reverse_array=reverse,
        groups=groups,
        n_splits=3,
    )

    assert len(result) == 3

    assert (
        result["group_overlap"] == 0
    ).all()

    assert (
        result["n_test_forward"]
        == result["n_test_reverse"]
    ).all()


def test_grouped_dv_evaluator_uses_expected_three_fold_structure():
    forward = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
        ],
        dtype=float,
    )

    reverse = -forward

    groups = np.arange(6)

    result = evaluate_grouped_dv_critic(
        forward_array=forward,
        reverse_array=reverse,
        groups=groups,
        n_splits=3,
    )

    np.testing.assert_array_equal(
        result["fold"].to_numpy(),
        np.array([1, 2, 3]),
    )

    assert (
        result["n_test_groups"] == 2
    ).all()


def test_identical_forward_reverse_distributions_give_zero_grouped_dv():
    forward = np.array(
        [
            [-1.0],
            [-0.5],
            [0.0],
            [0.5],
            [1.0],
            [1.5],
        ],
        dtype=float,
    )

    reverse = forward.copy()

    groups = np.arange(6)

    result = evaluate_grouped_dv_critic(
        forward_array=forward,
        reverse_array=reverse,
        groups=groups,
        n_splits=3,
    )

    np.testing.assert_allclose(
        result["dv_raw"].to_numpy(),
        np.zeros(3),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result["dv_clipped"].to_numpy(),
        np.zeros(3),
        atol=1e-12,
    )


def test_separable_forward_reverse_toy_data_give_positive_grouped_dv():
    forward = np.array(
        [
            [2.0],
            [2.1],
            [1.9],
            [2.2],
            [1.8],
            [2.05],
        ],
        dtype=float,
    )

    reverse = np.array(
        [
            [-2.0],
            [-2.1],
            [-1.9],
            [-2.2],
            [-1.8],
            [-2.05],
        ],
        dtype=float,
    )

    groups = np.arange(6)

    result = evaluate_grouped_dv_critic(
        forward_array=forward,
        reverse_array=reverse,
        groups=groups,
        n_splits=3,
    )

    assert (
        result["dv_raw"] > 0.0
    ).all()


def test_grouped_dv_evaluator_rejects_shape_mismatch():
    forward = np.ones(
        (6, 2)
    )

    reverse = np.ones(
        (5, 2)
    )

    groups = np.arange(6)

    with pytest.raises(ValueError):
        evaluate_grouped_dv_critic(
            forward_array=forward,
            reverse_array=reverse,
            groups=groups,
            n_splits=3,
        )


def test_grouped_dv_evaluator_rejects_too_few_groups():
    forward = np.ones(
        (4, 2)
    )

    reverse = -forward

    groups = np.array(
        [0, 0, 1, 1]
    )

    with pytest.raises(ValueError):
        evaluate_grouped_dv_critic(
            forward_array=forward,
            reverse_array=reverse,
            groups=groups,
            n_splits=3,
        )

def test_grouped_ou_path_samples_have_expected_shapes():
    forward, reverse, groups = (
        build_grouped_ou_path_samples(
            n_groups=6,
            n_steps_per_group=20,
            path_n_states=4,
            k=1.0,
            omega=1.0,
            diffusion=1.0,
            dt=0.1,
            seed=2031,
        )
    )

    # 21 states per trajectory -> 5 complete non-overlapping blocks.
    # 6 trajectories -> 30 samples.
    assert forward.shape == (30, 8)
    assert reverse.shape == (30, 8)
    assert groups.shape == (30,)

    assert np.unique(groups).size == 6


def test_grouped_ou_path_samples_are_reproducible():
    result_a = build_grouped_ou_path_samples(
        n_groups=6,
        n_steps_per_group=20,
        path_n_states=4,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
        seed=2031,
    )

    result_b = build_grouped_ou_path_samples(
        n_groups=6,
        n_steps_per_group=20,
        path_n_states=4,
        k=1.0,
        omega=1.0,
        diffusion=1.0,
        dt=0.1,
        seed=2031,
    )

    for array_a, array_b in zip(
        result_a,
        result_b,
    ):
        np.testing.assert_allclose(
            array_a,
            array_b,
        )


def test_grouped_ou_reverse_samples_are_exact_state_reversals():
    forward, reverse, _ = (
        build_grouped_ou_path_samples(
            n_groups=3,
            n_steps_per_group=12,
            path_n_states=4,
            k=1.0,
            omega=1.0,
            diffusion=1.0,
            dt=0.1,
            seed=2031,
        )
    )

    forward_paths = forward.reshape(
        -1,
        4,
        2,
    )

    reverse_paths = reverse.reshape(
        -1,
        4,
        2,
    )

    np.testing.assert_allclose(
        reverse_paths,
        forward_paths[:, ::-1, :],
    )


def test_summarize_grouped_dv_uses_weighted_raw_mean():
    fold_results = pd.DataFrame(
        {
            "n_test_forward": [
                10,
                20,
                30,
            ],
            "dv_raw": [
                0.1,
                -0.1,
                0.2,
            ],
        }
    )

    summary = summarize_grouped_dv(
        fold_results
    )

    expected_raw = np.average(
        np.array(
            [0.1, -0.1, 0.2]
        ),
        weights=np.array(
            [10, 20, 30]
        ),
    )

    assert summary["dv_raw"] == pytest.approx(
        expected_raw
    )

    assert summary[
        "dv_clipped"
    ] == pytest.approx(
        max(0.0, expected_raw)
    )


def test_summarize_grouped_dv_clips_only_after_averaging():
    fold_results = pd.DataFrame(
        {
            "n_test_forward": [
                1,
                1,
            ],
            "dv_raw": [
                0.1,
                -0.3,
            ],
        }
    )

    summary = summarize_grouped_dv(
        fold_results
    )

    assert summary["dv_raw"] == pytest.approx(
        -0.1
    )

    assert summary["dv_clipped"] == pytest.approx(
        0.0
    )


def test_grouped_ou_path_samples_reject_invalid_block_length():
    with pytest.raises(ValueError):
        build_grouped_ou_path_samples(
            n_groups=3,
            n_steps_per_group=2,
            path_n_states=10,
            k=1.0,
            omega=1.0,
            diffusion=1.0,
            dt=0.1,
            seed=2031,
        )

