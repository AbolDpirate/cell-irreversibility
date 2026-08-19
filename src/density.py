"""Reusable probability-density helpers for cell-displacement analysis."""

import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity


def get_displacement_series(
    data: pd.DataFrame,
    representation: str,
    tau_frames: int,
    component: str,
) -> pd.Series:
    """
    Return one displacement component for a selected
    representation and temporal lag.
    """

    valid_components = {
        "dx": "dx_um",
        "dy": "dy_um",
    }

    if component not in valid_components:
        raise ValueError(
            "component must be either 'dx' or 'dy'"
        )

    column_name = valid_components[component]

    mask = (
        data["representation"].eq(representation)
        & data["tau_frames"].eq(tau_frames)
    )

    return (
        data.loc[mask, column_name]
        .copy()
    )


def make_shared_bins(
    first: pd.Series,
    second: pd.Series,
    n_bins: int = 30,
) -> np.ndarray:
    """
    Build common histogram bin edges from two samples.
    """

    combined = pd.concat(
        [first, second],
        ignore_index=True,
    )

    minimum = combined.min()
    maximum = combined.max()

    return np.linspace(
        minimum,
        maximum,
        n_bins + 1,
    )


def freedman_diaconis_bin_width(
    values: pd.Series,
) -> float:
    """
    Calculate the Freedman-Diaconis histogram
    bin width for a one-dimensional sample.
    """

    clean_values = values.dropna()

    n = len(clean_values)

    if n < 2:
        raise ValueError(
            "At least two observations are required."
        )

    q25 = clean_values.quantile(0.25)
    q75 = clean_values.quantile(0.75)

    iqr = q75 - q25

    if iqr == 0:
        raise ValueError(
            "IQR is zero; Freedman-Diaconis "
            "bin width is undefined."
        )

    bin_width = (
        2
        * iqr
        / np.cbrt(n)
    )

    return float(bin_width)


def make_fd_bins(
    first: pd.Series,
    second: pd.Series,
) -> np.ndarray:
    """
    Build shared Freedman-Diaconis bin edges
    for two samples.
    """

    combined = pd.concat(
        [first, second],
        ignore_index=True,
    ).dropna()

    bin_width = (
        freedman_diaconis_bin_width(
            combined
        )
    )

    minimum = combined.min()
    maximum = combined.max()

    data_range = maximum - minimum

    n_bins = int(
        np.ceil(
            data_range / bin_width
        )
    )

    n_bins = max(n_bins, 1)

    return np.linspace(
        minimum,
        maximum,
        n_bins + 1,
    )


def silverman_bandwidth(
    values: pd.Series,
) -> float:
    """
    Estimate a robust one-dimensional KDE bandwidth.
    """

    clean_values = (
        values
        .dropna()
        .to_numpy()
    )

    n = len(clean_values)

    std = np.std(
        clean_values,
        ddof=1,
    )

    q25, q75 = np.quantile(
        clean_values,
        [0.25, 0.75],
    )

    iqr = q75 - q25

    robust_scale = min(
        std,
        iqr / 1.34,
    )

    bandwidth = (
        0.9
        * robust_scale
        * n ** (-1 / 5)
    )

    return float(bandwidth)


def evaluate_kde_1d(
    values: pd.Series,
    grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    Fit a one-dimensional Gaussian KDE and
    evaluate its density on a supplied grid.
    """

    sample = (
        values
        .dropna()
        .to_numpy()
        .reshape(-1, 1)
    )

    grid_2d = grid.reshape(-1, 1)

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=bandwidth,
    )

    kde.fit(sample)

    log_density = kde.score_samples(
        grid_2d
    )

    density = np.exp(
        log_density
    )

    return density


def get_displacement_points(
    data: pd.DataFrame,
    representation: str,
    tau_frames: int,
) -> np.ndarray:
    """
    Return two-dimensional displacement vectors
    [dx, dy] for one representation and lag.
    """

    mask = (
        data["representation"].eq(representation)
        & data["tau_frames"].eq(tau_frames)
    )

    points = (
        data
        .loc[
            mask,
            ["dx_um", "dy_um"],
        ]
        .dropna()
        .to_numpy()
    )

    return points


def robust_scale_1d(
    values: np.ndarray,
) -> float:
    """
    Estimate a robust characteristic scale
    for one coordinate.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    std = np.std(
        values,
        ddof=1,
    )

    q25, q75 = np.quantile(
        values,
        [0.25, 0.75],
    )

    iqr_scale = (
        (q75 - q25)
        / 1.34
    )

    valid_scales = [
        scale
        for scale in [std, iqr_scale]
        if np.isfinite(scale)
        and scale > 0
    ]

    if not valid_scales:
        raise ValueError(
            "Could not estimate a positive scale."
        )

    return min(valid_scales)


def shared_bandwidth_2d(
    raw_points: np.ndarray,
    comoving_points: np.ndarray,
) -> float:
    """
    Estimate one shared isotropic KDE bandwidth
    for raw and co-moving displacement data.
    """

    combined = np.vstack(
        [
            raw_points,
            comoving_points,
        ]
    )

    scale_x = robust_scale_1d(
        combined[:, 0]
    )

    scale_y = robust_scale_1d(
        combined[:, 1]
    )

    characteristic_scale = np.sqrt(
        scale_x * scale_y
    )

    n = min(
        len(raw_points),
        len(comoving_points),
    )

    bandwidth = (
        characteristic_scale
        * n ** (-1 / 6)
    )

    return float(bandwidth)


def make_symmetric_2d_grid(
    raw_points: np.ndarray,
    comoving_points: np.ndarray,
    bandwidth: float,
    n_grid: int = 101,
):
    """
    Construct a symmetric 2D evaluation grid
    around the displacement origin.
    """

    combined = np.vstack(
        [
            raw_points,
            comoving_points,
        ]
    )

    x_limit = (
        np.max(
            np.abs(
                combined[:, 0]
            )
        )
        + 3 * bandwidth
    )

    y_limit = (
        np.max(
            np.abs(
                combined[:, 1]
            )
        )
        + 3 * bandwidth
    )

    x_grid = np.linspace(
        -x_limit,
        x_limit,
        n_grid,
    )

    y_grid = np.linspace(
        -y_limit,
        y_limit,
        n_grid,
    )

    return x_grid, y_grid


def evaluate_kde_2d(
    points: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bandwidth: float,
):
    """
    Fit and evaluate a two-dimensional Gaussian KDE.
    """

    xx, yy = np.meshgrid(
        x_grid,
        y_grid,
    )

    evaluation_points = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
        ]
    )

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=bandwidth,
    )

    kde.fit(points)

    log_density = kde.score_samples(
        evaluation_points
    )

    density = (
        np.exp(log_density)
        .reshape(xx.shape)
    )

    return xx, yy, density