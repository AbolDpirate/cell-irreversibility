import numpy as np
import pandas as pd
import pytest

from src.sequences import (
    build_two_step_sequences,
    get_sequence_array,
    reverse_two_step_sequences,
    compute_two_step_features,
)


def test_build_two_step_sequences_requires_middle_frame():
    tracks = pd.DataFrame(
        {
            "cell_id": [1, 1],
            "frame": [0, 2],
            "t_min": [0.0, 40.0],
            "x_um": [0.0, 2.0],
            "y_um": [0.0, 0.0],
        }
    )

    sequences = build_two_step_sequences(tracks)

    assert len(sequences) == 0


def test_build_two_step_sequences_exact_continuity():
    tracks = pd.DataFrame(
        {
            "cell_id": [1, 1, 1, 1],
            "frame": [0, 1, 2, 3],
            "t_min": [0.0, 20.0, 40.0, 60.0],
            "x_um": [0.0, 1.0, 3.0, 6.0],
            "y_um": [0.0, 0.0, 1.0, 1.0],
        }
    )

    sequences = build_two_step_sequences(tracks)

    assert len(sequences) == 2

    assert sequences[
        "frame_start"
    ].tolist() == [0, 1]

    assert sequences[
        "frame_mid"
    ].tolist() == [1, 2]

    assert sequences[
        "frame_end"
    ].tolist() == [2, 3]

    assert (
        sequences["sequence_span_frames"] == 2
    ).all()

    assert np.allclose(
        sequences["sequence_span_min"],
        40.0,
    )


def test_get_sequence_array_preserves_column_order():
    sequences = pd.DataFrame(
        {
            "dx1_um": [1.0],
            "dy1_um": [2.0],
            "dx2_um": [3.0],
            "dy2_um": [4.0],
        }
    )

    sequence_array = get_sequence_array(
        sequences
    )

    assert sequence_array.shape == (1, 4)

    assert np.allclose(
        sequence_array[0],
        [1.0, 2.0, 3.0, 4.0],
    )


def test_reverse_two_step_sequences_is_involution():
    forward = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 1.0, 5.0, -3.0],
        ]
    )

    reversed_array = reverse_two_step_sequences(
        forward
    )

    expected = np.array(
        [
            [-3.0, -4.0, -1.0, -2.0],
            [-5.0, 3.0, 2.0, -1.0],
        ]
    )

    assert np.allclose(
        reversed_array,
        expected,
    )

    double_reversed = reverse_two_step_sequences(
        reversed_array
    )

    assert np.allclose(
        double_reversed,
        forward,
    )


def test_reverse_two_step_sequences_rejects_wrong_shape():
    with pytest.raises(ValueError):
        reverse_two_step_sequences(
            np.array([1.0, 2.0, 3.0])
        )


def test_two_step_features_change_sign_under_reversal():
    forward = np.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [1.0, 1.0, 2.0, 0.0],
        ]
    )

    reversed_array = reverse_two_step_sequences(
        forward
    )

    forward_features = compute_two_step_features(
        forward
    )

    reversed_features = compute_two_step_features(
        reversed_array
    )

    assert np.allclose(
        reversed_features[
            "delta_magnitude_um"
        ],
        -forward_features[
            "delta_magnitude_um"
        ],
    )

    valid = (
        forward_features["valid_turn"]
        & reversed_features["valid_turn"]
    )

    assert np.allclose(
        reversed_features.loc[
            valid,
            "turning_angle_rad",
        ],
        -forward_features.loc[
            valid,
            "turning_angle_rad",
        ],
    )


def test_get_scaled_feature_array_scales_without_centering():
    from src.sequences import get_scaled_feature_array

    features = pd.DataFrame(
        {
            "delta_magnitude_um": [2.0, -4.0],
            "signed_turn_sine": [0.5, -1.0],
        }
    )

    result = get_scaled_feature_array(
        features,
        delta_magnitude_scale=2.0,
        turn_sine_scale=0.5,
    )

    expected = np.array(
        [
            [1.0, 1.0],
            [-2.0, -2.0],
        ]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_sequence_kde_bandwidth_uses_two_dimensional_rule():
    from src.sequences import sequence_kde_bandwidth

    bandwidth = sequence_kde_bandwidth(
        365
    )

    assert np.isclose(
        bandwidth,
        365 ** (-1 / 6),
    )

    with pytest.raises(ValueError):
        sequence_kde_bandwidth(1)



def test_reversal_coordinate_decomposition():
    from src.sequences import (
        decompose_reversal_coordinates,
        reverse_two_step_sequences,
    )

    forward = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 1.0, 5.0, -3.0],
        ]
    )

    reversed_array = (
        reverse_two_step_sequences(
            forward
        )
    )

    forward_odd, forward_even = (
        decompose_reversal_coordinates(
            forward
        )
    )

    reversed_odd, reversed_even = (
        decompose_reversal_coordinates(
            reversed_array
        )
    )

    assert np.allclose(
        reversed_odd,
        -forward_odd,
    )

    assert np.allclose(
        reversed_even,
        forward_even,
    )


