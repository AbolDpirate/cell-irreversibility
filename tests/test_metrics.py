import numpy as np

from src.metrics import (
    inversion_asymmetry_index,
    compute_asymmetry_pair,
    bootstrap_cell_points,
)




def test_inversion_asymmetry_is_zero_for_symmetric_density():
    x_grid = np.array(
        [-1.0, 0.0, 1.0]
    )

    y_grid = np.array(
        [-1.0, 0.0, 1.0]
    )

    density = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )

    asymmetry, grid_mass = inversion_asymmetry_index(
        density=density,
        x_grid=x_grid,
        y_grid=y_grid,
    )

    assert np.isclose(
        asymmetry,
        0.0,
        atol=1e-12,
    )

    assert np.isfinite(grid_mass)
    assert grid_mass > 0


def test_inversion_asymmetry_is_positive_for_asymmetric_density():
    x_grid = np.array(
        [-1.0, 0.0, 1.0]
    )

    y_grid = np.array(
        [-1.0, 0.0, 1.0]
    )

    density = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 3.0, 8.0],
        ],
        dtype=float,
    )

    asymmetry, grid_mass = inversion_asymmetry_index(
        density=density,
        x_grid=x_grid,
        y_grid=y_grid,
    )

    assert np.isfinite(asymmetry)
    assert asymmetry > 0

    assert np.isfinite(grid_mass)
    assert grid_mass > 0


def test_inversion_asymmetry_is_unchanged_by_density_scaling():
    x_grid = np.array(
        [-1.0, 0.0, 1.0]
    )

    y_grid = np.array(
        [-1.0, 0.0, 1.0]
    )

    density = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
            [1.0, 4.0, 7.0],
        ],
        dtype=float,
    )

    asymmetry_1, _ = inversion_asymmetry_index(
        density=density,
        x_grid=x_grid,
        y_grid=y_grid,
    )

    asymmetry_2, _ = inversion_asymmetry_index(
        density=10.0 * density,
        x_grid=x_grid,
        y_grid=y_grid,
    )

    assert np.isclose(
        asymmetry_1,
        asymmetry_2,
        rtol=1e-12,
        atol=1e-12,
    )


def test_compute_asymmetry_pair_returns_two_valid_results():
    

    raw_points = np.array(
        [
            [-2.0, -1.0],
            [-1.0, -0.5],
            [0.0, 0.2],
            [1.0, 0.8],
            [2.0, 1.5],
        ]
    )

    comoving_points = np.array(
        [
            [-1.5, -0.8],
            [-0.7, -0.3],
            [0.0, 0.1],
            [0.8, 0.5],
            [1.4, 1.0],
        ]
    )

    result = compute_asymmetry_pair(
        raw_points=raw_points,
        comoving_points=comoving_points,
        tau_frames=1,
        n_grid=81,
    )

    assert len(result) == 2

    assert result["representation"].tolist() == [
        "raw",
        "comoving",
    ]

    assert (result["tau_frames"] == 1).all()

    assert np.isfinite(
        result["bandwidth_um"]
    ).all()

    assert np.isfinite(
        result["inversion_asymmetry_nats"]
    ).all()

    assert (
        result["inversion_asymmetry_nats"] >= 0
    ).all()

    assert np.allclose(
        result["bandwidth_um"].iloc[0],
        result["bandwidth_um"].iloc[1],
    )

    assert np.allclose(
        result["grid_mass"],
        1.0,
        atol=1e-3,
    )

def test_bootstrap_cell_points_preserves_paired_cell_blocks():
    import pandas as pd

    raw_tau = pd.DataFrame(
        {
            "cell_id": [1, 1, 2, 2],
            "dx_um": [1.0, 2.0, 10.0, 20.0],
            "dy_um": [3.0, 4.0, 30.0, 40.0],
        }
    )

    comoving_tau = pd.DataFrame(
        {
            "cell_id": [1, 1, 2, 2],
            "dx_um": [0.5, 1.5, 9.5, 19.5],
            "dy_um": [2.5, 3.5, 29.5, 39.5],
        }
    )

    rng = np.random.default_rng(2026)

    raw_points, comoving_points = (
        bootstrap_cell_points(
            raw_tau,
            comoving_tau,
            rng,
        )
    )

    assert raw_points.shape == comoving_points.shape
    assert raw_points.ndim == 2
    assert raw_points.shape[1] == 2
    assert len(raw_points) > 0
    assert np.isfinite(raw_points).all()
    assert np.isfinite(comoving_points).all()