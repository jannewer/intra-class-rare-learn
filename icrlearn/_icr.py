"""
This is a module for intra-class rarity estimators.
"""

# Authors: Janne Wernecken
# License: BSD 3 clause

from warnings import warn

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


# Note that the mixin class should always be on the left of `BaseEstimator` to ensure
# the MRO works as expected.
class ICRRandomForestClassifier(
    ClassifierMixin, BaseEstimator
):  # TODO: Maybe this can extend from BaseForest or BaseEnsemble?
    """A classifier that uses intra-class rarity.

    Parameters
    ----------
    rarity_measure : str, default='lof'
        The rarity measure to be used for the rarity score calculation.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

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

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "rarity_measure": [str],  # TODO: Add the list of valid values here
    }

    def __init__(self, rarity_measure="lof"):
        self.rarity_measure = rarity_measure

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba.

        Based on the implementation of the BaseForest (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py#L629)
        """
        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            accept_sparse=False,
            reset=False,
            ensure_all_finite=True,
        )
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based sparse matrices")
        return X

    def _validate_X_y_fit(self, X, y):
        """
        Validate X and y whenever fit is called.

        Input validation is based on the BaseForest implementation
        (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py#L329)
        """
        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
            accept_sparse=False,
            ensure_all_finite=True,
        )

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )

        return X, y

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fitting function for the ICRRandomForestClassifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        # Input validation is based on the BaseForest implementation
        # (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py#L329)
        X, y = self._validate_X_y_fit(X, y)

        # We need to make sure that we have a classification task
        check_classification_targets(y)

        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y

        # TODO: Implement actual fitting here

        # Return the classifier
        return self

    def predict(self, X):
        """Prediction function for the ICRRandomForestClassifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Input validation
        X = self._validate_X_predict(X)

        # TODO: Implement actual prediction here
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
