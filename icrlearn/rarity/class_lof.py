import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import _num_samples


def calculate_class_lof(X, y):
    rarity_scores = np.zeros(_num_samples(X))

    unique_classes = np.unique(y)
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]

        lof = LocalOutlierFactor()
        lof.fit_predict(X_class)
        lof_values_class = lof.negative_outlier_factor_
        # Invert the sign of the LOF values to get positive rarity scores
        lof_values_class = -lof_values_class

        rarity_scores[class_indices] = lof_values_class
    return rarity_scores
