"""Unit tests for displacement-step calculations."""

import pandas as pd
import pytest

from src.steps import (
    compute_steps_for_tau,
    compute_steps_multi_tau,
    summary_by_tau,
)


@pytest.fixture
def toy_tracks() -> pd.DataFrame:
    """Return two small trajectories with hand-checkable coordinates."""
    return pd.DataFrame(
        {
            "cell_id": [1, 1, 1, 2, 2, 2],
            "frame": [0, 1, 2, 0, 1, 2],
            "t_min": [0.0, 20.0, 40.0, 0.0, 20.0, 40.0],
            "x_um": [0.0, 1.0, 3.0, 10.0, 10.0, 9.0],
            "y_um": [0.0, 2.0, 2.0, 5.0, 6.0, 8.0],
        }
    )


def test_compute_steps_tau1(toy_tracks: pd.DataFrame) -> None:
    """τ=1 should produce the expected within-cell displacements."""
    steps = compute_steps_for_tau(toy_tracks, tau_frames=1)

    assert steps["cell_id"].tolist() == [1, 1, 2, 2]
    assert steps["frame_start"].tolist() == [0, 1, 0, 1]
    assert steps["frame_end"].tolist() == [1, 2, 1, 2]

    assert steps["dx_um"].tolist() == [1.0, 2.0, 0.0, -1.0]
    assert steps["dy_um"].tolist() == [2.0, 0.0, 1.0, 2.0]
    assert steps["tau_min"].tolist() == [20.0, 20.0, 20.0, 20.0]


def test_compute_steps_tau2(toy_tracks: pd.DataFrame) -> None:
    """τ=2 should compare frame 0 directly with frame 2."""
    steps = compute_steps_for_tau(toy_tracks, tau_frames=2)

    assert steps["cell_id"].tolist() == [1, 2]
    assert steps["frame_start"].tolist() == [0, 0]
    assert steps["frame_end"].tolist() == [2, 2]

    assert steps["dx_um"].tolist() == [3.0, -1.0]
    assert steps["dy_um"].tolist() == [2.0, 3.0]
    assert steps["tau_min"].tolist() == [40.0, 40.0]


def test_steps_do_not_cross_cell_boundaries(
    toy_tracks: pd.DataFrame,
) -> None:
    """The last point of one cell must never connect to another cell."""
    steps = compute_steps_for_tau(toy_tracks, tau_frames=1)

    assert len(steps) == 4
    assert steps.groupby("cell_id").size().to_dict() == {1: 2, 2: 2}


def test_invalid_tau_raises_error(toy_tracks: pd.DataFrame) -> None:
    """A lag smaller than one frame is physically and computationally invalid."""
    with pytest.raises(ValueError, match="tau_frames must be >= 1"):
        compute_steps_for_tau(toy_tracks, tau_frames=0)


def test_missing_required_column_raises_error(
    toy_tracks: pd.DataFrame,
) -> None:
    """The function should fail clearly when an essential column is absent."""
    incomplete_tracks = toy_tracks.drop(columns="y_um")

    with pytest.raises(ValueError, match="missing required columns"):
        compute_steps_for_tau(incomplete_tracks, tau_frames=1)


def test_multi_tau_summary(toy_tracks: pd.DataFrame) -> None:
    """Combined τ values should produce a correct summary table."""
    all_steps = compute_steps_multi_tau(toy_tracks, taus=[1, 2])
    summary = summary_by_tau(all_steps)

    assert summary["tau_frames"].tolist() == [1, 2]
    assert summary["n_steps"].tolist() == [4, 2]
    assert summary["n_cells"].tolist() == [2, 2]
    assert summary["median_tau_min"].tolist() == [20.0, 40.0]


def test_missing_frame_is_not_mislabeled_as_tau1() -> None:
    """
    Frames 0 and 2 are two physical frames apart.

    They must not be treated as a tau=1 displacement merely because
    they are adjacent rows in the DataFrame.
    """
    tracks_with_gap = pd.DataFrame(
        {
            "cell_id": [1, 1],
            "frame": [0, 2],
            "t_min": [0.0, 40.0],
            "x_um": [0.0, 3.0],
            "y_um": [0.0, 1.0],
        }
    )

    tau1_steps = compute_steps_for_tau(
        tracks_with_gap,
        tau_frames=1,
    )

    assert tau1_steps.empty

    tau2_steps = compute_steps_for_tau(
        tracks_with_gap,
        tau_frames=2,
    )

    assert len(tau2_steps) == 1
    assert tau2_steps["frame_start"].tolist() == [0]
    assert tau2_steps["frame_end"].tolist() == [2]
    assert tau2_steps["dx_um"].tolist() == [3.0]
    assert tau2_steps["dy_um"].tolist() == [1.0]
    assert tau2_steps["tau_min"].tolist() == [40.0]


def test_duplicate_cell_frame_raises_error() -> None:
    """One cell cannot have two observations in the same frame."""
    duplicate_tracks = pd.DataFrame(
        {
            "cell_id": [1, 1],
            "frame": [0, 0],
            "t_min": [0.0, 0.0],
            "x_um": [1.0, 1.5],
            "y_um": [2.0, 2.5],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate cell_id-frame observations",
    ):
        compute_steps_for_tau(
            duplicate_tracks,
            tau_frames=1,
        )