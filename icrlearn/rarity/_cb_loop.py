import numpy as np
import pandas as pd
from PyNomaly import loop
from sklearn.utils.validation import _num_samples


def calculate_cb_loop(X, y):
    if isinstance(X, pd.DataFrame):
        X = X.values

    rarity_scores = np.zeros(_num_samples(X))

    unique_classes = np.unique(y)
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]

        # Handle classes with only one sample as very rare
        if len(class_indices) < 2:
            rarity_scores[class_indices] = 1
            continue

        fitted_loop = loop.LocalOutlierProbability(X_class).fit()
        loop_values_class = fitted_loop.local_outlier_probabilities

        rarity_scores[class_indices] = loop_values_class
    return rarity_scores
