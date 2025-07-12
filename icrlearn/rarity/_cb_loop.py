import sys
import timeit

import numpy as np
import pandas as pd
from PyNomaly import loop
from sklearn.utils.validation import _num_samples


def calculate_cb_loop(X, y, min_score=0.5, extent=3, n_neighbors=10, timing=False):
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

        # Scale the loop values to the range [min_score, 2 * min_score]
        loop_values_class = min_score * (loop_values_class + 1)
        rarity_scores[class_indices] = loop_values_class

        if timing:
            end_time = timeit.default_timer()
            print(
                f"CB-LoOP: Time taken for class {class_label}:"
                f" {end_time - start_time:.4f} seconds"
            )
    return rarity_scores
