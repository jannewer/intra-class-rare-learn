"""
This is a module for intra-class rarity estimators.
"""

# Authors: Janne Wernecken
# License: BSD 3 clause

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree._tree import DTYPE, issparse
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    _num_samples,
    check_is_fitted,
    get_tags,
    validate_data,
)


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

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        rarity_measure="lof",
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
        self.rarity_measure = rarity_measure

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = False
        tags.input_tags.allow_nan = self.rarity_measure != "lof"
        return tags

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
                rarity_scores = np.zeros(_num_samples(X))

                unique_classes = np.unique(y)
                for class_label in unique_classes:
                    class_indices = np.where(y == class_label)[0]
                    X_class = X[class_indices]

                    lof = LocalOutlierFactor()
                    lof.fit_predict(X_class)
                    lof_values_class = lof.negative_outlier_factor_
                    # Invert the sign of the LOF values to get a positive rarity scores
                    lof_values_class = -lof_values_class

                    rarity_scores[class_indices] = lof_values_class

                return rarity_scores
            case _:
                raise ValueError(f"Unknown rarity measure: {self.rarity_measure}")

    def fit(self, X, y, sample_weight=None):
        ensure_all_finite = "allow-nan" if get_tags(self).input_tags.allow_nan else True
        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            ensure_all_finite=ensure_all_finite,
        )
        check_classification_targets(y)

        rarity_scores = self.calculate_rarity_scores(X, y)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight) * np.asarray(rarity_scores)
        else:
            sample_weight = np.asarray(rarity_scores)

        super().fit(X, y, sample_weight)

        return self

    def _validate_X_predict(self, X):
        """Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)

        if (get_tags(self).input_tags.allow_nan) & (
            self.estimators_[0]._support_missing_values(X)
        ):
            ensure_all_finite = "allow-nan"
        else:
            ensure_all_finite = True

        X = validate_data(
            self,
            X,
            dtype=DTYPE,
            accept_sparse="csr",
            reset=False,
            ensure_all_finite=ensure_all_finite,
        )
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based sparse matrices")
        return X
