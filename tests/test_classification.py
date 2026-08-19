import numpy as np
import pytest

from src.classification import (
    build_classification_dataset,
    randomize_pair_orientation,
    empirical_upper_pvalue,
    evaluate_grouped_classifier,
)


def test_build_classification_dataset_is_balanced_and_grouped():
    forward = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    reversed_array = np.array(
        [
            [-1.0, -2.0],
            [-3.0, -4.0],
        ]
    )

    cell_ids = np.array(
        [10, 20]
    )

    X, y, groups = (
        build_classification_dataset(
            forward,
            reversed_array,
            cell_ids,
        )
    )

    assert X.shape == (4, 2)

    assert y.tolist() == [
        1, 1,
        0, 0,
    ]

    assert groups.tolist() == [
        10, 20,
        10, 20,
    ]

    assert np.allclose(
        X[:2],
        forward,
    )

    assert np.allclose(
        X[2:],
        reversed_array,
    )


def test_build_classification_dataset_rejects_mismatch():
    forward = np.zeros(
        (2, 4)
    )

    reversed_array = np.zeros(
        (3, 4)
    )

    cell_ids = np.array(
        [1, 2]
    )

    with pytest.raises(ValueError):
        build_classification_dataset(
            forward,
            reversed_array,
            cell_ids,
        )


def test_randomize_pair_orientation_preserves_pairs():
    forward = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    reversed_array = np.array(
        [
            [-1.0, -2.0],
            [-3.0, -4.0],
            [-5.0, -6.0],
        ]
    )

    swap_mask = np.array(
        [False, True, False]
    )

    pseudo_forward, pseudo_reverse = (
        randomize_pair_orientation(
            forward,
            reversed_array,
            swap_mask,
        )
    )

    assert np.allclose(
        pseudo_forward[0],
        forward[0],
    )

    assert np.allclose(
        pseudo_reverse[0],
        reversed_array[0],
    )

    assert np.allclose(
        pseudo_forward[1],
        reversed_array[1],
    )

    assert np.allclose(
        pseudo_reverse[1],
        forward[1],
    )


def test_empirical_upper_pvalue_uses_plus_one_correction():
    null_values = np.array(
        [0.1, 0.2, 0.3, 0.4]
    )

    p_value = empirical_upper_pvalue(
        observed=0.35,
        null_values=null_values,
    )

    assert np.isclose(
        p_value,
        2 / 5,
    )


def test_evaluate_grouped_classifier_preserves_groups():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array(
        [
            [2.0, 1.0],
            [-2.0, -1.0],
            [3.0, 0.5],
            [-3.0, -0.5],
            [1.5, 2.0],
            [-1.5, -2.0],
            [2.5, 1.5],
            [-2.5, -1.5],
            [1.0, 3.0],
            [-1.0, -3.0],
            [3.0, 2.0],
            [-3.0, -2.0],
        ]
    )

    y = np.array(
        [
            1, 0,
            1, 0,
            1, 0,
            1, 0,
            1, 0,
            1, 0,
        ]
    )

    groups = np.array(
        [
            1, 1,
            2, 2,
            3, 3,
            4, 4,
            5, 5,
            6, 6,
        ]
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000
                ),
            ),
        ]
    )

    result = evaluate_grouped_classifier(
        model=model,
        X=X,
        y=y,
        groups=groups,
        cv=GroupKFold(
            n_splits=3
        ),
    )

    assert len(result) == 3

    assert result[
        "roc_auc"
    ].between(
        0.0,
        1.0,
    ).all()

    assert result[
        "balanced_accuracy"
    ].between(
        0.0,
        1.0,
    ).all()


