import sys
import timeit

import numpy as np
import pandas as pd
from PyNomaly import loop
from sklearn.utils.validation import _num_samples


def calculate_cb_loop(X, y, min_score=0.0, extent=3, n_neighbors=10, timing=False):
    """Calculate Class-Based Local Outlier Probability (CB-LoOP) rarity scores.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples. Internally, its dtype will be converted
        to ``dtype=np.float32``. If a sparse matrix is provided, it will be
        converted into a sparse ``csc_matrix``.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The class labels of the input samples.

    min_score : float, default=0.0
        The minimum rarity score to assign to samples that are not rare.
        Samples with a rarity score smaller than min_score will be set to `min_score`.

    extent : int, default=3
        The extent of the local neighborhood to consider.
        See :func:`PyNomaly.loop.LocalOutlierProbability` for more details.

    n_neighbors : int, default=10
        The number of neighbors to consider.

    timing : bool, default=False
        If True, prints the time taken for processing each class.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The rarity scores for each sample in `X`.
        If `min_score` is set to 0.0, the scores will be in the range [0, 1],
        where 0 indicates a common sample and 1 indicates a rare sample.

    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    use_numba = "numba" in sys.modules
    rarity_scores = np.zeros(_num_samples(X))

    unique_classes = np.unique(y)
    for class_label in unique_classes:
        start_time = None
        if timing:
            start_time = timeit.default_timer()
            print(f"CB-LoOP: Processing class {class_label}...")

        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]

        # Handle classes with only one sample as very rare
        if len(class_indices) < 2:
            rarity_scores[class_indices] = 1
            continue

        fitted_loop = loop.LocalOutlierProbability(
            X_class, extent=extent, n_neighbors=n_neighbors, use_numba=use_numba
        ).fit()
        loop_values_class = fitted_loop.local_outlier_probabilities

        if min_score == 0.0:
            # If min_score is 0, return the loop values directly
            rarity_scores[class_indices] = loop_values_class

            if timing:
                end_time = timeit.default_timer()
                print(
                    f"CB-LoOP: Time taken for class {class_label}:"
                    f" {end_time - start_time:.4f} seconds"
                )

            continue

        # Scale the loop values to the range [min_score, 1]
        loop_values_class[loop_values_class <= min_score] = min_score
        rarity_scores[class_indices] = loop_values_class

        if timing:
            end_time = timeit.default_timer()
            print(
                f"CB-LoOP: Time taken for class {class_label}:"
                f" {end_time - start_time:.4f} seconds"
            )
    return rarity_scores
