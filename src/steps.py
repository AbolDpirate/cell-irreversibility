# src/steps.py

from __future__ import annotations
import pandas as pd
import numpy as np


REQUIRED_TRACK_COLS = {"cell_id", "frame", "t_min", "x_um", "y_um"}




def validate_tracks(tracks: pd.DataFrame) -> None:
    """
    Validate the cleaned trajectory table.

    Required columns:
        cell_id, frame, t_min, x_um, y_um

    Each cell must have at most one observation in each frame.
    """
    missing = REQUIRED_TRACK_COLS - set(tracks.columns)

    if missing:
        raise ValueError(
            f"tracks is missing required columns: {sorted(missing)}"
        )

    duplicate_mask = tracks.duplicated(
        subset=["cell_id", "frame"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_pairs = (
            tracks.loc[duplicate_mask, ["cell_id", "frame"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )

        raise ValueError(
            "tracks contains duplicate cell_id-frame observations. "
            f"Examples: {duplicate_pairs}"
        )


def compute_steps_for_tau(
    tracks: pd.DataFrame,
    tau_frames: int,
) -> pd.DataFrame:
    """
    Compute displacement steps for an exact physical frame lag.

    A step is included only when the same cell has observations at both:

        frame_start
        frame_start + tau_frames

    Missing intermediate frames do not cause the lag to be mislabeled.
    """
    validate_tracks(tracks)

    if tau_frames < 1:
        raise ValueError("tau_frames must be >= 1")

    tracks_sorted = (
        tracks
        .sort_values(["cell_id", "frame"])
        .reset_index(drop=True)
        .copy()
    )

    starts = (
        tracks_sorted
        .rename(
            columns={
                "frame": "frame_start",
                "t_min": "t_start_min",
                "x_um": "x_start_um",
                "y_um": "y_start_um",
            }
        )
        [
            [
                "cell_id",
                "frame_start",
                "t_start_min",
                "x_start_um",
                "y_start_um",
            ]
        ]
        .copy()
    )

    # Define the exact target frame.
    starts["frame_end"] = (
        starts["frame_start"] + int(tau_frames)
    )

    ends = (
        tracks_sorted
        .rename(
            columns={
                "frame": "frame_end",
                "t_min": "t_end_min",
                "x_um": "x_end_um",
                "y_um": "y_end_um",
            }
        )
        [
            [
                "cell_id",
                "frame_end",
                "t_end_min",
                "x_end_um",
                "y_end_um",
            ]
        ]
        .copy()
    )

    # Match each starting observation to the same cell at the exact
    # requested future frame.
    steps = starts.merge(
        ends,
        on=["cell_id", "frame_end"],
        how="inner",
        validate="one_to_one",
    )

    steps["dx_um"] = (
        steps["x_end_um"] - steps["x_start_um"]
    )

    steps["dy_um"] = (
        steps["y_end_um"] - steps["y_start_um"]
    )

    steps["tau_frames"] = int(tau_frames)

    steps["tau_min"] = (
        steps["t_end_min"] - steps["t_start_min"]
    )

    steps["cell_id"] = steps["cell_id"].astype(int)
    steps["frame_start"] = steps["frame_start"].astype(int)
    steps["frame_end"] = steps["frame_end"].astype(int)

    output_columns = [
        "cell_id",
        "frame_start",
        "t_start_min",
        "frame_end",
        "t_end_min",
        "dx_um",
        "dy_um",
        "tau_frames",
        "tau_min",
    ]

    return (
        steps[output_columns]
        .sort_values(["cell_id", "frame_start"])
        .reset_index(drop=True)
    )

def compute_steps_multi_tau(tracks: pd.DataFrame, taus: list[int]) -> pd.DataFrame:
    """
    Compute steps for multiple taus and concatenate into one DataFrame.
    The output includes 'tau_frames' and 'tau_min' so you can filter later.
    """
    validate_tracks(tracks)

    all_steps = []
    for tau in taus:
        s = compute_steps_for_tau(tracks, tau_frames=tau)
        all_steps.append(s)

    steps_all = pd.concat(all_steps, ignore_index=True)
    return steps_all


def summary_by_tau(steps_all: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact summary table per tau:
      - number of steps
      - number of contributing cell_ids
      - median tau_min
    """
    summary = (steps_all.groupby("tau_frames")
                        .agg(
                            n_steps=("dx_um", "size"),
                            n_cells=("cell_id", "nunique"),
                            median_tau_min=("tau_min", "median"),
                        )
                        .reset_index()
              )
    return summary


def compute_framewise_common_motion(
    tracks,
):
    """Estimate frame-wise population common motion.

    Exact one-frame displacements are grouped by physical frame
    transition. Common dx and dy are the component-wise medians
    across contributing cells.
    """
    one_frame_steps = compute_steps_for_tau(
        tracks,
        tau_frames=1,
    )

    common_motion = (
        one_frame_steps
        .groupby(
            [
                "frame_start",
                "frame_end",
            ],
            as_index=False,
        )
        .agg(
            common_dx_um=(
                "dx_um",
                "median",
            ),
            common_dy_um=(
                "dy_um",
                "median",
            ),
            n_cells=(
                "cell_id",
                "nunique",
            ),
        )
        .sort_values(
            "frame_start"
        )
        .reset_index(
            drop=True
        )
    )

    return common_motion


def make_comoving_tracks(
    tracks,
    common_motion,
):
    """Subtract cumulative frame-wise population common motion.

    Frame 0 receives zero cumulative correction.

    The returned table preserves:
        cell_id
        frame
        t_min
        x_um
        y_um
    """
    frame_correction = pd.DataFrame(
        {
            "frame": [0],
            "common_x_um": [0.0],
            "common_y_um": [0.0],
        }
    )

    transition_correction = pd.DataFrame(
        {
            "frame":
                common_motion[
                    "frame_end"
                ].to_numpy(),

            "common_x_um":
                np.cumsum(
                    common_motion[
                        "common_dx_um"
                    ].to_numpy()
                ),

            "common_y_um":
                np.cumsum(
                    common_motion[
                        "common_dy_um"
                    ].to_numpy()
                ),
        }
    )

    frame_correction = pd.concat(
        [
            frame_correction,
            transition_correction,
        ],
        ignore_index=True,
    )

    result = (
        tracks
        .merge(
            frame_correction,
            on="frame",
            how="left",
            validate="many_to_one",
        )
        .copy()
    )

    if (
        result[
            [
                "common_x_um",
                "common_y_um",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Missing cumulative common-motion "
            "correction for one or more frames."
        )

    result["x_um"] = (
        result["x_um"]
        - result["common_x_um"]
    )

    result["y_um"] = (
        result["y_um"]
        - result["common_y_um"]
    )

    return (
        result[
            [
                "cell_id",
                "frame",
                "t_min",
                "x_um",
                "y_um",
            ]
        ]
        .sort_values(
            [
                "cell_id",
                "frame",
            ]
        )
        .reset_index(
            drop=True
        )
    )

