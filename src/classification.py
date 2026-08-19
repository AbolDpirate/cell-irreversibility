"""Reusable helpers for forward-versus-reversed classification."""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


def build_classification_dataset(
    forward_array,
    reverse_array,
    cell_ids,
):
    """
    Build a balanced forward-versus-reversed
    classification dataset.
    """

    if forward_array.shape != reverse_array.shape:
        raise ValueError(
            "Forward and reverse arrays must have identical shapes."
        )

    if len(cell_ids) != len(forward_array):
        raise ValueError(
            "cell_ids length does not match the number of sequences."
        )

    X = np.vstack(
        [
            forward_array,
            reverse_array,
        ]
    )

    y = np.concatenate(
        [
            np.ones(
                len(forward_array),
                dtype=int,
            ),
            np.zeros(
                len(reverse_array),
                dtype=int,
            ),
        ]
    )

    groups = np.concatenate(
        [
            cell_ids,
            cell_ids,
        ]
    )

    return X, y, groups


def randomize_pair_orientation(
    forward_array,
    reverse_array,
    swap_mask,
):
    """
    Randomly exchange temporal orientation within
    exact forward/reverse sequence pairs.
    """

    pseudo_forward = forward_array.copy()
    pseudo_reverse = reverse_array.copy()

    pseudo_forward[
        swap_mask
    ] = reverse_array[
        swap_mask
    ]

    pseudo_reverse[
        swap_mask
    ] = forward_array[
        swap_mask
    ]

    return (
        pseudo_forward,
        pseudo_reverse,
    )


def empirical_upper_pvalue(
    observed,
    null_values,
):
    """
    Finite-simulation upper-tail probability
    with the standard +1 correction.
    """

    null_values = np.asarray(
        null_values,
        dtype=float,
    )

    return (
        1
        + np.sum(
            null_values >= observed
        )
    ) / (
        len(null_values)
        + 1
    )


def evaluate_grouped_classifier(
    model,
    X,
    y,
    groups,
    cv,
) -> pd.DataFrame:
    """
    Evaluate a classifier using group-preserving
    cross-validation.

    All preprocessing contained in the supplied
    model/Pipeline is fitted separately inside
    each training fold.
    """

    records = []

    for fold_id, (
        train_indices,
        test_indices,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        train_groups = np.unique(
            groups[train_indices]
        )

        test_groups = np.unique(
            groups[test_indices]
        )

        overlap = np.intersect1d(
            train_groups,
            test_groups,
        )

        if len(overlap) != 0:
            raise RuntimeError(
                "Group leakage detected."
            )

        model.fit(
            X[train_indices],
            y[train_indices],
        )

        if hasattr(
            model,
            "predict_proba",
        ):
            scores = model.predict_proba(
                X[test_indices]
            )[:, 1]

        elif hasattr(
            model,
            "decision_function",
        ):
            scores = model.decision_function(
                X[test_indices]
            )

        else:
            raise TypeError(
                "Model must expose predict_proba "
                "or decision_function."
            )

        predictions = model.predict(
            X[test_indices]
        )

        roc_auc = roc_auc_score(
            y[test_indices],
            scores,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y[test_indices],
                predictions,
            )
        )

        records.append(
            {
                "fold": fold_id,
                "n_train_examples": len(
                    train_indices
                ),
                "n_test_examples": len(
                    test_indices
                ),
                "n_train_groups": len(
                    train_groups
                ),
                "n_test_groups": len(
                    test_groups
                ),
                "roc_auc": float(
                    roc_auc
                ),
                "balanced_accuracy": float(
                    balanced_accuracy
                ),
            }
        )

    return pd.DataFrame(records)

