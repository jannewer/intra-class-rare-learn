import pandas as pd
from sklearn.neighbors import NearestNeighbors


def calculate_l2min(X, y, n_neighbors=5, psi=1, beta=0.5):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

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
    scaled_proportion_other_classes = (proportion_other_classes + 1) * beta

    return scaled_proportion_other_classes
