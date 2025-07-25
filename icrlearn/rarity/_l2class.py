import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def calculate_l2class(X, y, n_neighbors=5, psi=1, beta=0.5):
    """Calculate L2Class rarity scores.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples. Internally, its dtype will be converted
        to ``dtype=np.float32``. If a sparse matrix is provided, it will be
        converted into a sparse ``csc_matrix``.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The class labels of the input samples.

    n_neighbors : int, default=5
        The number of neighbors to consider for the rarity score calculation.

    psi : float, default=1
        The exponent to scale the count of other classes.
        The default of 1 equates to a linear scaling.

    beta : float, default=0.5
        The minimum rarity score to assign to samples that are not rare.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The rarity scores for each sample in `X`.
        If `beta` is set to 0.0, the scores will be in the range [0, 1],
        where 0 indicates a common sample and 1 indicates a rare sample.

    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    y = np.asarray(y)

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    ).fit(X)

    knn_distances, knn_indices = nn.kneighbors(X)

    count_other_classes = (y[knn_indices] != y[:, None]).sum(axis=1)
    scaled_count_other_classes = count_other_classes**psi
    proportion_other_classes = scaled_count_other_classes / n_neighbors

    if beta == 0.0:
        return proportion_other_classes

    scaled_proportion_other_classes = (proportion_other_classes + 1) * beta
    return scaled_proportion_other_classes
