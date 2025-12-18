# src/steps.py

from __future__ import annotations
import pandas as pd


REQUIRED_TRACK_COLS = {"cell_id", "frame", "t_min", "x_um", "y_um"}


def validate_tracks(tracks: pd.DataFrame) -> None:
    """
    Validate that the input tracks DataFrame has the required columns.
    Raises ValueError with a clear message if something is missing.
    """
    missing = REQUIRED_TRACK_COLS - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks is missing required columns: {sorted(missing)}")


def compute_steps_for_tau(tracks: pd.DataFrame, tau_frames: int) -> pd.DataFrame:
    """
    Compute displacement steps Δ for a given lag tau_frames.

    Input 'tracks' must have columns:
      cell_id (int), frame (int), t_min (float), x_um (float), y_um (float)

    Returns a DataFrame with one row per step:
      cell_id, frame_start, t_start_min, frame_end, t_end_min, dx_um, dy_um,
      tau_frames, tau_min
    """
    validate_tracks(tracks)

    if tau_frames < 1:
        raise ValueError("tau_frames must be >= 1")

    # Sort to ensure correct temporal order inside each cell_id
    tracks_sorted = (
        tracks.sort_values(["cell_id", "frame"])
              .reset_index(drop=True)
              .copy()
    )

    # Group by trajectory (cell_id) and use shift to align t+tau next to t
    g = tracks_sorted.groupby("cell_id", group_keys=False)

    x_future = g["x_um"].shift(-tau_frames)
    y_future = g["y_um"].shift(-tau_frames)
    t_future = g["t_min"].shift(-tau_frames)
    f_future = g["frame"].shift(-tau_frames)

    # Displacements
    dx = x_future - tracks_sorted["x_um"]
    dy = y_future - tracks_sorted["y_um"]

    steps = pd.DataFrame({
        "cell_id": tracks_sorted["cell_id"].astype(int),
        "frame_start": tracks_sorted["frame"].astype(int),
        "t_start_min": tracks_sorted["t_min"].astype(float),
        "frame_end": f_future,
        "t_end_min": t_future,
        "dx_um": dx,
        "dy_um": dy,
        "tau_frames": int(tau_frames),
    })

    # tau in minutes (computed from timestamps, robust if time step is not constant)
    steps["tau_min"] = steps["t_end_min"] - steps["t_start_min"]

    # Remove rows where the future point does not exist (end of each track)
    steps = steps.dropna(subset=["dx_um", "dy_um", "frame_end", "t_end_min"]).copy()

    # Fix dtypes after dropna
    steps["frame_end"] = steps["frame_end"].astype(int)
    steps["t_end_min"] = steps["t_end_min"].astype(float)

    return steps


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
