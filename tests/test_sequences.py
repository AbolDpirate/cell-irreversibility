import numpy as np
import pandas as pd
import pytest

from src.sequences import (
    build_two_step_sequences,
    get_sequence_array,
    reverse_two_step_sequences,
    compute_two_step_features,
    build_k_step_sequences,
    get_k_step_sequence_array,
    reverse_k_step_sequences,
    decompose_three_step_parity,
    reconstruct_three_step_from_parity,
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


def test_build_k_step_sequences_three_step_exact_continuity():
    tracks = pd.DataFrame(
        {
            "cell_id": [
                1,
                1,
                1,
                1,
                1,
            ],
            "frame": [
                0,
                1,
                2,
                3,
                4,
            ],
            "t_min": [
                0.0,
                20.0,
                40.0,
                60.0,
                80.0,
            ],
            "x_um": [
                0.0,
                1.0,
                3.0,
                6.0,
                10.0,
            ],
            "y_um": [
                0.0,
                0.0,
                1.0,
                1.0,
                2.0,
            ],
        }
    )

    sequences = (
        build_k_step_sequences(
            tracks,
            n_steps=3,
        )
    )

    assert len(sequences) == 2

    assert (
        sequences[
            "frame_start"
        ].tolist()
        == [0, 1]
    )

    assert (
        sequences[
            "frame_1"
        ].tolist()
        == [1, 2]
    )

    assert (
        sequences[
            "frame_2"
        ].tolist()
        == [2, 3]
    )

    assert (
        sequences[
            "frame_end"
        ].tolist()
        == [3, 4]
    )

    assert (
        sequences[
            "sequence_span_frames"
        ]
        == 3
    ).all()

    assert np.allclose(
        sequences[
            "sequence_span_min"
        ],
        60.0,
    )


def test_build_k_step_sequences_rejects_missing_intermediate_frame():
    tracks = pd.DataFrame(
        {
            "cell_id": [
                1,
                1,
                1,
                1,
            ],
            "frame": [
                0,
                1,
                3,
                4,
            ],
            "t_min": [
                0.0,
                20.0,
                60.0,
                80.0,
            ],
            "x_um": [
                0.0,
                1.0,
                3.0,
                4.0,
            ],
            "y_um": [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    sequences = (
        build_k_step_sequences(
            tracks,
            n_steps=3,
        )
    )

    assert len(sequences) == 0


def test_get_k_step_sequence_array_preserves_order():
    sequences = pd.DataFrame(
        {
            "dx1_um": [1.0],
            "dy1_um": [2.0],
            "dx2_um": [3.0],
            "dy2_um": [4.0],
            "dx3_um": [5.0],
            "dy3_um": [6.0],
        }
    )

    result = (
        get_k_step_sequence_array(
            sequences,
            n_steps=3,
        )
    )

    assert result.shape == (
        1,
        6,
    )

    assert np.allclose(
        result[0],
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ],
    )


def test_reverse_k_step_sequences_three_step_is_exact_involution():
    forward = np.array(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            [
                -1.0,
                3.0,
                2.0,
                -4.0,
                7.0,
                1.0,
            ],
        ]
    )

    reversed_array = (
        reverse_k_step_sequences(
            forward
        )
    )

    expected = np.array(
        [
            [
                -5.0,
                -6.0,
                -3.0,
                -4.0,
                -1.0,
                -2.0,
            ],
            [
                -7.0,
                -1.0,
                -2.0,
                4.0,
                1.0,
                -3.0,
            ],
        ]
    )

    assert np.allclose(
        reversed_array,
        expected,
    )

    double_reversed = (
        reverse_k_step_sequences(
            reversed_array
        )
    )

    assert np.allclose(
        double_reversed,
        forward,
    )


def test_k_step_helpers_reject_invalid_inputs():
    tracks = pd.DataFrame(
        {
            "cell_id": [1],
            "frame": [0],
            "t_min": [0.0],
            "x_um": [0.0],
            "y_um": [0.0],
        }
    )

    with pytest.raises(
        ValueError
    ):
        build_k_step_sequences(
            tracks,
            n_steps=0,
        )

    with pytest.raises(
        ValueError
    ):
        build_k_step_sequences(
            tracks,
            n_steps=2.5,
        )

    with pytest.raises(
        ValueError
    ):
        reverse_k_step_sequences(
            np.array(
                [
                    [
                        1.0,
                        2.0,
                        3.0,
                    ]
                ]
            )
        )

def test_generic_two_step_builder_matches_historical_builder():
    tracks = pd.DataFrame(
        {
            "cell_id": [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
            ],
            "frame": [
                0,
                1,
                2,
                3,
                0,
                1,
                3,
            ],
            "t_min": [
                0.0,
                20.0,
                40.0,
                60.0,
                0.0,
                20.0,
                60.0,
            ],
            "x_um": [
                0.0,
                1.0,
                3.0,
                6.0,
                10.0,
                11.0,
                15.0,
            ],
            "y_um": [
                0.0,
                0.5,
                1.0,
                2.0,
                5.0,
                6.0,
                8.0,
            ],
        }
    )

    historical = (
        build_two_step_sequences(
            tracks
        )
    )

    generic = (
        build_k_step_sequences(
            tracks,
            n_steps=2,
        )
    )

    historical_array = (
        get_sequence_array(
            historical
        )
    )

    generic_array = (
        get_k_step_sequence_array(
            generic,
            n_steps=2,
        )
    )

    assert len(generic) == len(
        historical
    )

    assert np.array_equal(
        generic["cell_id"].to_numpy(),
        historical["cell_id"].to_numpy(),
    )

    assert np.array_equal(
        generic["frame_start"].to_numpy(),
        historical["frame_start"].to_numpy(),
    )

    assert np.array_equal(
        generic["frame_end"].to_numpy(),
        historical["frame_end"].to_numpy(),
    )

    assert np.allclose(
        generic_array,
        historical_array,
    )

def test_three_step_parity_coordinates_match_manual_values():
    forward = np.array(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]
        ]
    )

    (
        net_odd,
        internal_odd,
        reversal_even,
    ) = decompose_three_step_parity(
        forward
    )

    assert np.allclose(
        net_odd,
        [
            [
                9.0,
                12.0,
            ]
        ],
    )

    assert np.allclose(
        internal_odd,
        [
            [
                0.0,
                0.0,
            ]
        ],
    )

    assert np.allclose(
        reversal_even,
        [
            [
                -4.0,
                -4.0,
            ]
        ],
    )

def test_three_step_parity_transforms_correctly_under_reversal():
    forward = np.array(
        [
            [
                1.0,
                2.0,
                3.0,
                -1.0,
                4.0,
                5.0,
            ],
            [
                -2.0,
                1.0,
                0.5,
                3.0,
                7.0,
                -4.0,
            ],
        ]
    )

    reversed_array = (
        reverse_k_step_sequences(
            forward
        )
    )

    (
        n_forward,
        q_forward,
        e_forward,
    ) = decompose_three_step_parity(
        forward
    )

    (
        n_reverse,
        q_reverse,
        e_reverse,
    ) = decompose_three_step_parity(
        reversed_array
    )

    assert np.allclose(
        n_reverse,
        -n_forward,
    )

    assert np.allclose(
        q_reverse,
        -q_forward,
    )

    assert np.allclose(
        e_reverse,
        e_forward,
    )

def test_three_step_parity_decomposition_is_invertible():
    forward = np.array(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            [
                -2.0,
                0.5,
                7.0,
                -3.0,
                1.5,
                9.0,
            ],
        ]
    )

    (
        net_odd,
        internal_odd,
        reversal_even,
    ) = decompose_three_step_parity(
        forward
    )

    reconstructed = (
        reconstruct_three_step_from_parity(
            net_odd,
            internal_odd,
            reversal_even,
        )
    )

    assert reconstructed.shape == (
        2,
        6,
    )

    assert np.allclose(
        reconstructed,
        forward,
    )

def test_three_step_parity_helpers_reject_invalid_shapes():
    with pytest.raises(
        ValueError
    ):
        decompose_three_step_parity(
            np.array(
                [
                    [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                    ]
                ]
            )
        )

    with pytest.raises(
        ValueError
    ):
        reconstruct_three_step_from_parity(
            np.zeros((3, 2)),
            np.zeros((4, 2)),
            np.zeros((3, 2)),
        )

    with pytest.raises(
        ValueError
    ):
        reconstruct_three_step_from_parity(
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
        )

def test_three_step_net_odd_equals_endpoint_displacement():
    tracks = pd.DataFrame(
        {
            "cell_id": [
                1,
                1,
                1,
                1,
            ],
            "frame": [
                0,
                1,
                2,
                3,
            ],
            "t_min": [
                0.0,
                20.0,
                40.0,
                60.0,
            ],
            "x_um": [
                2.0,
                3.5,
                6.0,
                10.0,
            ],
            "y_um": [
                -1.0,
                0.0,
                2.0,
                5.0,
            ],
        }
    )

    sequences = (
        build_k_step_sequences(
            tracks,
            n_steps=3,
        )
    )

    sequence_array = (
        get_k_step_sequence_array(
            sequences,
            n_steps=3,
        )
    )

    (
        net_odd,
        internal_odd,
        reversal_even,
    ) = decompose_three_step_parity(
        sequence_array
    )

    expected_endpoint_displacement = np.array(
        [
            [
                10.0 - 2.0,
                5.0 - (-1.0),
            ]
        ]
    )

    assert np.allclose(
        net_odd,
        expected_endpoint_displacement,
    )

