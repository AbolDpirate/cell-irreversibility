"""Reusable statistical metrics for cell-motility analysis."""

import numpy as np
import pandas as pd

from src.density import (
    evaluate_kde_2d,
    make_symmetric_2d_grid,
    shared_bandwidth_2d,
)



def inversion_asymmetry_index(
    density: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
):
    """
    Compare a normalized 2D density with its
    inversion-reflected counterpart.
    """

    dx_grid = (
        x_grid[1]
        - x_grid[0]
    )

    dy_grid = (
        y_grid[1]
        - y_grid[0]
    )

    cell_area = (
        dx_grid
        * dy_grid
    )

    grid_mass = (
        density.sum()
        * cell_area
    )

    p = (
        density
        / grid_mass
    )

    p_reversed = (
        p[::-1, ::-1]
    )

    epsilon = (
        p.max()
        * 1e-12
    )

    p_safe = (
        p + epsilon
    )

    reversed_safe = (
        p_reversed + epsilon
    )

    p_safe = (
        p_safe
        / (
            p_safe.sum()
            * cell_area
        )
    )

    reversed_safe = (
        reversed_safe
        / (
            reversed_safe.sum()
            * cell_area
        )
    )

    log_ratio = np.log(
        p_safe
        / reversed_safe
    )

    asymmetry = (
        (
            p_safe
            * log_ratio
        ).sum()
        * cell_area
    )

    return float(asymmetry), float(grid_mass)


def compute_asymmetry_pair(
    raw_points: np.ndarray,
    comoving_points: np.ndarray,
    tau_frames: int,
    bandwidth_multiplier: float = 1.0,
    n_grid: int = 101,
) -> pd.DataFrame:
    """
    Calculate raw and co-moving inversion asymmetry
    using a shared KDE bandwidth and symmetric grid.
    """

    baseline_bandwidth = shared_bandwidth_2d(
        raw_points,
        comoving_points,
    )

    bandwidth = (
        baseline_bandwidth
        * bandwidth_multiplier
    )

    x_grid, y_grid = make_symmetric_2d_grid(
        raw_points,
        comoving_points,
        bandwidth=bandwidth,
        n_grid=n_grid,
    )

    records = []

    for representation, points in [
        ("raw", raw_points),
        ("comoving", comoving_points),
    ]:

        _, _, density = evaluate_kde_2d(
            points,
            x_grid,
            y_grid,
            bandwidth,
        )

        asymmetry, grid_mass = (
            inversion_asymmetry_index(
                density,
                x_grid,
                y_grid,
            )
        )

        records.append(
            {
                "tau_frames": tau_frames,
                "representation": representation,
                "bandwidth_multiplier": bandwidth_multiplier,
                "bandwidth_um": bandwidth,
                "grid_mass": grid_mass,
                "inversion_asymmetry_nats": asymmetry,
            }
        )

    return pd.DataFrame(records)


def bootstrap_cell_points(
    raw_tau: pd.DataFrame,
    comoving_tau: pd.DataFrame,
    rng: np.random.Generator,
):
    """
    Resample cells with replacement and return paired
    raw and co-moving displacement point arrays.
    """

    raw_cells = set(
        raw_tau["cell_id"].unique()
    )

    comoving_cells = set(
        comoving_tau["cell_id"].unique()
    )

    common_cells = sorted(
        raw_cells.intersection(
            comoving_cells
        )
    )

    sampled_cells = rng.choice(
        common_cells,
        size=len(common_cells),
        replace=True,
    )

    raw_blocks = []
    comoving_blocks = []

    for cell_id in sampled_cells:

        raw_block = (
            raw_tau
            .loc[
                raw_tau["cell_id"].eq(cell_id),
                ["dx_um", "dy_um"],
            ]
            .to_numpy()
        )

        comoving_block = (
            comoving_tau
            .loc[
                comoving_tau["cell_id"].eq(cell_id),
                ["dx_um", "dy_um"],
            ]
            .to_numpy()
        )

        raw_blocks.append(
            raw_block
        )

        comoving_blocks.append(
            comoving_block
        )

    raw_points = np.vstack(
        raw_blocks
    )

    comoving_points = np.vstack(
        comoving_blocks
    )

    return (
        raw_points,
        comoving_points,
    )
