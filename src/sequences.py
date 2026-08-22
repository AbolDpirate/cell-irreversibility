"""Reusable helpers for ordered cell-motility sequences."""

import numpy as np
import pandas as pd

from src.steps import (
    validate_tracks,
    compute_steps_for_tau,
)


SEQUENCE_COMPONENT_COLUMNS = [
    "dx1_um",
    "dy1_um",
    "dx2_um",
    "dy2_um",
]


def build_two_step_sequences(
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct exact two-step displacement sequences.

    Each retained sequence contains:

        frame_start
            ->
        frame_mid = frame_start + 1
            ->
        frame_end = frame_start + 2

    Therefore, every sequence contains two genuinely
    consecutive one-frame displacement steps.
    """

    validate_tracks(tracks)

    one_frame_steps = compute_steps_for_tau(
        tracks,
        tau_frames=1,
    )

    first_steps = (
        one_frame_steps
        .rename(
            columns={
                "frame_end": "frame_mid",
                "t_end_min": "t_mid_min",
                "dx_um": "dx1_um",
                "dy_um": "dy1_um",
            }
        )
        [
            [
                "cell_id",
                "frame_start",
                "t_start_min",
                "frame_mid",
                "t_mid_min",
                "dx1_um",
                "dy1_um",
            ]
        ]
        .copy()
    )

    second_steps = (
        one_frame_steps
        .rename(
            columns={
                "frame_start": "frame_mid",
                "dx_um": "dx2_um",
                "dy_um": "dy2_um",
            }
        )
        [
            [
                "cell_id",
                "frame_mid",
                "frame_end",
                "t_end_min",
                "dx2_um",
                "dy2_um",
            ]
        ]
        .copy()
    )

    sequences = first_steps.merge(
        second_steps,
        on=[
            "cell_id",
            "frame_mid",
        ],
        how="inner",
        validate="one_to_one",
    )

    sequences[
        "sequence_span_frames"
    ] = (
        sequences["frame_end"]
        - sequences["frame_start"]
    )

    sequences[
        "sequence_span_min"
    ] = (
        sequences["t_end_min"]
        - sequences["t_start_min"]
    )

    valid_first_transition = (
        sequences["frame_mid"]
        == sequences["frame_start"] + 1
    )

    valid_second_transition = (
        sequences["frame_end"]
        == sequences["frame_mid"] + 1
    )

    if not (
        valid_first_transition.all()
        and valid_second_transition.all()
    ):
        raise RuntimeError(
            "Non-consecutive frames detected "
            "inside a two-step sequence."
        )

    return (
        sequences
        .sort_values(
            [
                "cell_id",
                "frame_start",
            ]
        )
        .reset_index(drop=True)
    )


def get_sequence_array(
    sequences: pd.DataFrame,
) -> np.ndarray:
    """
    Return two-step sequences as an (n, 4) NumPy array.

    Column order:
        dx1, dy1, dx2, dy2
    """

    return (
        sequences[
            SEQUENCE_COMPONENT_COLUMNS
        ]
        .to_numpy(dtype=float)
    )


def reverse_two_step_sequences(
    sequence_array: np.ndarray,
) -> np.ndarray:
    """
    Apply the two-step time-reversal operator.

    Forward:
        [dx1, dy1, dx2, dy2]

    Reversed:
        [-dx2, -dy2, -dx1, -dy1]
    """

    sequence_array = np.asarray(
        sequence_array,
        dtype=float,
    )

    if (
        sequence_array.ndim != 2
        or sequence_array.shape[1] != 4
    ):
        raise ValueError(
            "Expected an array with shape (n, 4)."
        )

    reversed_array = np.column_stack(
        [
            -sequence_array[:, 2],
            -sequence_array[:, 3],
            -sequence_array[:, 0],
            -sequence_array[:, 1],
        ]
    )

    return reversed_array


def compute_two_step_features(
    sequence_array: np.ndarray,
) -> pd.DataFrame:
    """
    Compute interpretable time-odd features
    for two-step displacement sequences.
    """

    sequence_array = np.asarray(
        sequence_array,
        dtype=float,
    )

    if (
        sequence_array.ndim != 2
        or sequence_array.shape[1] != 4
    ):
        raise ValueError(
            "Expected an array with shape (n, 4)."
        )

    dx1 = sequence_array[:, 0]
    dy1 = sequence_array[:, 1]
    dx2 = sequence_array[:, 2]
    dy2 = sequence_array[:, 3]

    magnitude1 = np.hypot(
        dx1,
        dy1,
    )

    magnitude2 = np.hypot(
        dx2,
        dy2,
    )

    delta_magnitude = (
        magnitude2
        - magnitude1
    )

    cross_term = (
        dx1 * dy2
        - dy1 * dx2
    )

    dot_term = (
        dx1 * dx2
        + dy1 * dy2
    )

    turning_angle_rad = np.arctan2(
        cross_term,
        dot_term,
    )

    valid_turn = (
        (magnitude1 > 0)
        & (magnitude2 > 0)
    )

    turning_angle_rad = np.where(
        valid_turn,
        turning_angle_rad,
        np.nan,
    )

    return pd.DataFrame(
        {
            "magnitude1_um": magnitude1,
            "magnitude2_um": magnitude2,
            "delta_magnitude_um": delta_magnitude,
            "turning_angle_rad": turning_angle_rad,
            "turning_angle_deg": np.degrees(
                turning_angle_rad
            ),
            "valid_turn": valid_turn,
        }
    )


def get_scaled_feature_array(
    feature_table: pd.DataFrame,
    delta_magnitude_scale: float,
    turn_sine_scale: float,
) -> np.ndarray:
    """
    Convert the two selected time-odd features
    into a dimensionless scaled array.

    No mean centering is performed because
    the time-reversal origin must remain at zero.
    """

    delta_magnitude_scaled = (
        feature_table[
            "delta_magnitude_um"
        ].to_numpy()
        / delta_magnitude_scale
    )

    turn_sine_scaled = (
        feature_table[
            "signed_turn_sine"
        ].to_numpy()
        / turn_sine_scale
    )

    return np.column_stack(
        [
            delta_magnitude_scaled,
            turn_sine_scaled,
        ]
    )


def sequence_kde_bandwidth(
    n_samples: int,
) -> float:
    """
    Two-dimensional rule-of-thumb bandwidth
    after feature scaling.
    """

    if n_samples < 2:
        raise ValueError(
            "At least two samples are required."
        )

    return float(
        n_samples ** (-1 / 6)
    )


def decompose_reversal_coordinates(
    sequence_array: np.ndarray,
):
    """
    Decompose a two-step sequence into
    reversal-odd and reversal-even coordinates.

    odd  = d1 + d2
    even = d1 - d2
    """

    sequence_array = np.asarray(
        sequence_array,
        dtype=float,
    )

    if (
        sequence_array.ndim != 2
        or sequence_array.shape[1] != 4
    ):
        raise ValueError(
            "Expected an array with shape (n, 4)."
        )

    first_step = sequence_array[:, 0:2]
    second_step = sequence_array[:, 2:4]

    odd = (
        first_step
        + second_step
    )

    even = (
        first_step
        - second_step
    )

    return odd, even

def _validate_n_steps(
    n_steps,
) -> int:
    """
    Validate and normalize a requested sequence length.
    """

    if (
        isinstance(n_steps, bool)
        or not isinstance(
            n_steps,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "n_steps must be an integer >= 1."
        )

    n_steps = int(n_steps)

    if n_steps < 1:
        raise ValueError(
            "n_steps must be an integer >= 1."
        )

    return n_steps


def build_k_step_sequences(
    tracks: pd.DataFrame,
    n_steps: int,
) -> pd.DataFrame:
    """
    Construct exact ordered displacement sequences
    containing an arbitrary number of one-frame steps.

    An n-step sequence requires observations at

        t, t+1, ..., t+n

    for the same cell.

    Missing intermediate frames therefore prevent
    construction of the sequence.
    """

    validate_tracks(tracks)

    n_steps = _validate_n_steps(
        n_steps
    )

    one_frame_steps = (
        compute_steps_for_tau(
            tracks,
            tau_frames=1,
        )
    )

    def frame_column(
        position,
    ):
        if position == 0:
            return "frame_start"

        if position == n_steps:
            return "frame_end"

        return f"frame_{position}"

    def time_column(
        position,
    ):
        if position == 0:
            return "t_start_min"

        if position == n_steps:
            return "t_end_min"

        return f"t_{position}_min"

    first_frame_column = (
        frame_column(1)
    )

    first_time_column = (
        time_column(1)
    )

    sequences = (
        one_frame_steps
        .rename(
            columns={
                "frame_end":
                    first_frame_column,

                "t_end_min":
                    first_time_column,

                "dx_um":
                    "dx1_um",

                "dy_um":
                    "dy1_um",
            }
        )
        [
            [
                "cell_id",
                "frame_start",
                "t_start_min",
                first_frame_column,
                first_time_column,
                "dx1_um",
                "dy1_um",
            ]
        ]
        .copy()
    )

    for step_index in range(
        2,
        n_steps + 1,
    ):

        previous_frame_column = (
            frame_column(
                step_index - 1
            )
        )

        current_frame_column = (
            frame_column(
                step_index
            )
        )

        current_time_column = (
            time_column(
                step_index
            )
        )

        next_steps = (
            one_frame_steps
            .rename(
                columns={
                    "frame_start":
                        previous_frame_column,

                    "frame_end":
                        current_frame_column,

                    "t_end_min":
                        current_time_column,

                    "dx_um":
                        f"dx{step_index}_um",

                    "dy_um":
                        f"dy{step_index}_um",
                }
            )
            [
                [
                    "cell_id",
                    previous_frame_column,
                    current_frame_column,
                    current_time_column,
                    f"dx{step_index}_um",
                    f"dy{step_index}_um",
                ]
            ]
            .copy()
        )

        sequences = sequences.merge(
            next_steps,
            on=[
                "cell_id",
                previous_frame_column,
            ],
            how="inner",
            validate="one_to_one",
        )

    for position in range(
        1,
        n_steps + 1,
    ):

        expected_frame = (
            sequences[
                "frame_start"
            ]
            + position
        )

        observed_frame = (
            sequences[
                frame_column(
                    position
                )
            ]
        )

        if not (
            observed_frame
            == expected_frame
        ).all():
            raise RuntimeError(
                "Non-consecutive frames detected "
                "inside a k-step sequence."
            )

    sequences[
        "sequence_span_frames"
    ] = (
        sequences["frame_end"]
        - sequences["frame_start"]
    )

    sequences[
        "sequence_span_min"
    ] = (
        sequences["t_end_min"]
        - sequences["t_start_min"]
    )

    metadata_columns = [
        "cell_id",
        "frame_start",
        "t_start_min",
    ]

    for position in range(
        1,
        n_steps,
    ):
        metadata_columns.extend(
            [
                frame_column(
                    position
                ),
                time_column(
                    position
                ),
            ]
        )

    metadata_columns.extend(
        [
            "frame_end",
            "t_end_min",
        ]
    )

    component_columns = []

    for step_index in range(
        1,
        n_steps + 1,
    ):
        component_columns.extend(
            [
                f"dx{step_index}_um",
                f"dy{step_index}_um",
            ]
        )

    output_columns = (
        metadata_columns
        + component_columns
        + [
            "sequence_span_frames",
            "sequence_span_min",
        ]
    )

    return (
        sequences[
            output_columns
        ]
        .sort_values(
            [
                "cell_id",
                "frame_start",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def get_k_step_sequence_array(
    sequences: pd.DataFrame,
    n_steps: int,
) -> np.ndarray:
    """
    Convert an n-step sequence table into an
    (n_sequences, 2 * n_steps) NumPy array.

    Column order is

        dx1, dy1, dx2, dy2, ..., dxn, dyn.
    """

    n_steps = _validate_n_steps(
        n_steps
    )

    component_columns = []

    for step_index in range(
        1,
        n_steps + 1,
    ):
        component_columns.extend(
            [
                f"dx{step_index}_um",
                f"dy{step_index}_um",
            ]
        )

    missing_columns = (
        set(component_columns)
        - set(sequences.columns)
    )

    if missing_columns:
        raise ValueError(
            "Sequence table is missing "
            f"components: "
            f"{sorted(missing_columns)}"
        )

    return (
        sequences[
            component_columns
        ]
        .to_numpy(
            dtype=float
        )
    )


def reverse_k_step_sequences(
    sequence_array: np.ndarray,
) -> np.ndarray:
    """
    Apply exact time reversal to an arbitrary
    ordered displacement sequence.

    Forward:

        (d1, d2, ..., dn)

    Reversed:

        (-dn, ..., -d2, -d1)
    """

    sequence_array = np.asarray(
        sequence_array,
        dtype=float,
    )

    if (
        sequence_array.ndim != 2
        or sequence_array.shape[1] < 2
        or sequence_array.shape[1] % 2 != 0
    ):
        raise ValueError(
            "Expected a two-dimensional array "
            "with an even number of columns."
        )

    step_vectors = (
        sequence_array.reshape(
            sequence_array.shape[0],
            -1,
            2,
        )
    )

    reversed_vectors = (
        -step_vectors[
            :,
            ::-1,
            :
        ]
    )

    return (
        reversed_vectors.reshape(
            sequence_array.shape
        )
    )