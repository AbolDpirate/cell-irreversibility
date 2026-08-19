import numpy as np
import pandas as pd
import pytest

from src.density import (
    get_displacement_series,
    freedman_diaconis_bin_width,
    silverman_bandwidth,
    robust_scale_1d,
    shared_bandwidth_2d,
    make_symmetric_2d_grid,
    evaluate_kde_2d,
)


def test_get_displacement_series_selects_requested_subset():
    data = pd.DataFrame(
        {
            "representation": ["raw", "raw", "comoving", "raw"],
            "tau_frames": [1, 2, 1, 1],
            "dx_um": [1.0, 2.0, 3.0, 4.0],
            "dy_um": [-1.0, -2.0, -3.0, -4.0],
        }
    )

    result = get_displacement_series(
        data=data,
        representation="raw",
        tau_frames=1,
        component="dx",
    )

    assert result.tolist() == [1.0, 4.0]


def test_get_displacement_series_rejects_invalid_component():
    data = pd.DataFrame(
        {
            "representation": ["raw"],
            "tau_frames": [1],
            "dx_um": [1.0],
            "dy_um": [2.0],
        }
    )

    with pytest.raises(ValueError):
        get_displacement_series(
            data=data,
            representation="raw",
            tau_frames=1,
            component="dz",
        )


def test_freedman_diaconis_bin_width_is_positive():
    values = pd.Series(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    )

    width = freedman_diaconis_bin_width(values)

    assert np.isfinite(width)
    assert width > 0


def test_silverman_bandwidth_is_positive():
    values = pd.Series(
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    )

    bandwidth = silverman_bandwidth(values)

    assert np.isfinite(bandwidth)
    assert bandwidth > 0


def test_robust_scale_1d_is_positive():
    values = np.array(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    scale = robust_scale_1d(values)

    assert np.isfinite(scale)
    assert scale > 0


def test_shared_bandwidth_2d_is_positive():
    raw_points = np.array(
        [
            [-1.0, -0.5],
            [0.0, 0.2],
            [1.0, 0.8],
            [2.0, 1.2],
        ]
    )

    comoving_points = np.array(
        [
            [-0.8, -0.4],
            [0.1, 0.1],
            [0.9, 0.7],
            [1.8, 1.0],
        ]
    )

    bandwidth = shared_bandwidth_2d(
        raw_points,
        comoving_points,
    )

    assert np.isfinite(bandwidth)
    assert bandwidth > 0


def test_make_symmetric_2d_grid_is_centered_on_zero():
    raw_points = np.array(
        [
            [-2.0, -1.0],
            [1.0, 3.0],
        ]
    )

    comoving_points = np.array(
        [
            [-1.0, -2.0],
            [2.0, 1.0],
        ]
    )

    x_grid, y_grid = make_symmetric_2d_grid(
        raw_points,
        comoving_points,
        bandwidth=0.5,
        n_grid=101,
    )

    assert np.allclose(
        x_grid,
        -x_grid[::-1],
    )

    assert np.allclose(
        y_grid,
        -y_grid[::-1],
    )

    assert np.isclose(
        x_grid[len(x_grid) // 2],
        0.0,
    )

    assert np.isclose(
        y_grid[len(y_grid) // 2],
        0.0,
    )


def test_evaluate_kde_2d_returns_valid_density():
    points = np.array(
        [
            [-1.0, -1.0],
            [-0.5, 0.0],
            [0.5, 0.0],
            [1.0, 1.0],
        ]
    )

    x_grid = np.linspace(
        -4.0,
        4.0,
        101,
    )

    y_grid = np.linspace(
        -4.0,
        4.0,
        101,
    )

    xx, yy, density = evaluate_kde_2d(
        points=points,
        x_grid=x_grid,
        y_grid=y_grid,
        bandwidth=0.5,
    )

    assert xx.shape == (101, 101)
    assert yy.shape == (101, 101)
    assert density.shape == (101, 101)

    assert np.isfinite(density).all()
    assert (density >= 0).all()
    assert density.sum() > 0