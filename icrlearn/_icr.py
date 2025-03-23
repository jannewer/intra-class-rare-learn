"""
This is a module for intra-class rarity estimators.
"""

# Authors: Janne Wernecken
# License: BSD 3 clause

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor


class ICRRandomForestClassifier(RandomForestClassifier):
    """A RF classifier that uses intra-class rarity.
    Based on scikit-learn's RandomForestClassifier.

    Parameters
    ----------
    rarity_measure : str, default='lof'
        The rarity measure to be used for the rarity score calculation.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from icrlearn import ICRRandomForestClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> icr_rf = ICRRandomForestClassifier().fit(X, y)
    >>> icr_rf.predict(X)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """

    def __init__(self, rarity_measure="lof"):
        super().__init__()
        self.rarity_measure = rarity_measure

    def calculate_rarity_scores(self, X, y):
        """
        Calculate rarity scores for each sample in the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The class labels of the input samples.

        Returns
        -------

        rarity_scores : array-like of shape (n_samples,)
            The rarity scores for each input sample.

        """

        match self.rarity_measure:
            case "lof":
                # TODO: Implement the class-specific LOF calculation here
                clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                # Calculate LOF scores
                clf.fit_predict(X)
                # Get the degree of abnormality of each sample
                # (higher values ~= normal ~= more likely to be inlier ~= less rare)
                neg_lof = clf.negative_outlier_factor_
                # Invert LOF scores to get a rarity score (higher values ~= more rare)
                pos_lof = -neg_lof

                return pos_lof
            case _:
                raise ValueError(f"Unknown rarity measure: {self.rarity_measure}")

    def fit(self, X, y, sample_weight=None):
        rarity_scores = self.calculate_rarity_scores(X, y)

        if sample_weight is not None:
            sample_weight = sample_weight * rarity_scores
        else:
            sample_weight = rarity_scores

        super().fit(X, y, sample_weight)

        return self
