"""
This is a module for intra-class rarity estimators.
"""

# Authors: Janne Wernecken
# License: BSD 3 clause

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor


class ICRRandomForestClassifier(RandomForestClassifier):
    # TODO: Extend docs
    """A classifier that uses intra-class rarity.
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

    # TODO: Check how to validate the rarity measure parameter
    # e.g. Using _parameter_constraints?

    def __init__(self, rarity_measure="lof"):
        super().__init__()
        self.rarity_measure = rarity_measure

    def calculate_rarity_scores(self, X, y):
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
            # TODO: Is this the best way to combine sample weights and rarity scores?
            sample_weight = sample_weight * rarity_scores
        else:
            sample_weight = rarity_scores

        # TODO: Think about other ways to use the rarity scores in the fitting process
        # E.g. by adjusting the bootstrap sampling:
        # --> overwrite BaseForest._get_n_samples_bootstrap()
        # --> or set _n_samples_bootstrap directly
        super().fit(X, y, sample_weight)
