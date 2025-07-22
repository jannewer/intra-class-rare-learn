"""
This is a module for intra-class rarity estimators.
"""

# Authors: Janne Wernecken
# License: BSD 3 clause

from warnings import catch_warnings, simplefilter, warn

import numpy as np
from sklearn.base import (
    is_classifier,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import MAX_INT, _get_n_samples_bootstrap
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import robust_scale
from sklearn.tree._tree import DOUBLE, DTYPE, issparse
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils._tags import get_tags
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

from icrlearn.rarity import calculate_cb_loop
from icrlearn.rarity._l2class import calculate_l2class


def _generate_sample_indices(
    random_state, n_samples, n_samples_bootstrap, rarity_scores
):
    """
    Private function used in _parallel_build_trees to generate sample indices.

    Widely based on the original _generate_sample_indices function from sklearn.
    It has been modified to include rarity scores.
    They are used to bias sampling probabilities towards intra-class rare samples.
    """

    if np.sum(rarity_scores) == 0:
        # Fall back to uniform sampling if all rarity scores are 0 (no rare samples)
        return np.random.randint(0, n_samples, size=n_samples_bootstrap)

    # Normalize rarity scores to make them sum to 1 (to use them as probabilities)
    rarity_scores = rarity_scores / np.sum(rarity_scores)

    # Draw a random subsample of indices with replacement
    # Use the rarity scores as probabilities for sampling to give intra-class rare
    # samples a higher chance of being selected
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.choice(
        np.arange(n_samples),
        size=n_samples_bootstrap,
        replace=True,
        p=rarity_scores,
    )

    return sample_indices


def _parallel_build_trees_rarity_oversampling(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    rarity_scores,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
    missing_values_in_feature_mask=None,
):
    """
    Private function used to fit a single tree in parallel.

    Widely based on the original _parallel_build_trees function from sklearn.
    It has been modified to include intra-class rarity scores for sampling.
    """

    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if bootstrap:
        n_samples = X.shape[0]

        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        # This is the only change to the original sklearn _parallel_build_trees function
        # It is needed to pass the rarity_scores to the bootstrap sampling
        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap, rarity_scores
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        tree._fit(
            X,
            y,
            sample_weight=curr_sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )
    else:
        tree._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )
    return tree


class ICRRandomForestClassifier(RandomForestClassifier):
    """A RF classifier that uses intra-class rarity.
    Based on scikit-learn's RandomForestClassifier.

    Parameters
    ----------
    rarity_measure : {"cb_loop", "l2class"}, default="cb_loop"
        The rarity measure to be used for the rarity score calculation.
        Supported values are "cb_loop" to use the CB-LoOP algorithm or "l2class" to use
        the adapted L^2_min algorithm to calculate rarity scores.

    rarity_adjustment_method : {"bootstrap_sampling", "sample_weights"}, default="bootstrap_sampling"
        The method to bias the model towards intra-class rare samples during training.
        Supported values are "bootstrap_sampling" to use the rarity scores to adjust the
        bootstrap sampling process or "sample_weights" to use the rarity scores to
        weight the samples in the trees of the forest instead.

    n_neighbors : int, default=None
        The number of neighbors to consider for the rarity score calculation.
        If None, defaults to 10 for "cb_loop" and 5 for "l2class".

    min_rarity_score : float, default=0.0
        The minimum rarity score to assign to samples that are not rare.

    cb_loop_extent : int, default=3
        The extent parameter for the CB-LoOP algorithm. Only used if
        `rarity_measure` is set to "cb_loop". This parameter controls the
        sensitivity of the scoring.
        See `PyNomaly documentation <https://github.com/vc1492a/PyNomaly?tab=readme-ov-file#choosing-parameters>`__ for more details.

    l2class_psi : float, default=1
        The psi parameter for the L^2_min algorithm. Only used if
        `rarity_measure` is set to "l2class". This parameter controls the
        scaling of the count of other classes in the neighborhood.
        The default of 1 equates to a linear scaling.

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
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """  # noqa: E501

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
        rarity_measure="cb_loop",
        rarity_adjustment_method="bootstrap_sampling",
        n_neighbors=None,
        min_rarity_score=0.0,
        cb_loop_extent=3,
        l2class_psi=1,
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
        self.rarity_adjustment_method = rarity_adjustment_method
        self.n_neighbors = n_neighbors
        self.min_rarity_score = min_rarity_score
        self.cb_loop_extent = cb_loop_extent
        self.l2class_psi = l2class_psi

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = False
        tags.input_tags.allow_nan = (
            self.rarity_measure != "cb_loop" and self.rarity_measure != "l2class"
        )
        tags.input_tags.sparse = (
            self.rarity_measure != "cb_loop" and self.rarity_measure != "l2class"
        )
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

        X_scaled = robust_scale(X)

        match self.rarity_measure:
            case "cb_loop":
                n_neighbors_cb_loop = self.n_neighbors if self.n_neighbors else 10
                return calculate_cb_loop(
                    X_scaled,
                    y,
                    min_score=self.min_rarity_score,
                    extent=self.cb_loop_extent,
                    n_neighbors=n_neighbors_cb_loop,
                )
            case "l2class":
                n_neighbors_l2class = self.n_neighbors if self.n_neighbors else 5
                scores = calculate_l2class(
                    X_scaled,
                    y,
                    n_neighbors=n_neighbors_l2class,
                    psi=self.l2class_psi,
                    beta=self.min_rarity_score,
                )

                # in case of multi-output, take the mean across outputs
                # this is to ensure compatibility with the RandomForestClassifier
                # as the sample_weight parameter expects a 1D array
                if scores.ndim > 1:
                    return np.mean(scores, axis=1)

                return scores
            case _:
                raise ValueError(f"Unknown rarity measure: {self.rarity_measure}")

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y), adjusting for
        intra-class rarity.

        This method calculates rarity scores for the training data and uses them
        to adjust the sample weights or bootstrap sampling during the fitting
        process. If `rarity_adjustment_method` is set to "sample_weights", the
        rarity scores are used to weight the samples directly. If set to
        "bootstrap_sampling", the rarity scores are used to adjust the
        bootstrap sampling process.

        Widely based on the original fit method of sklearn's RandomForestClassifier,
        only modified to include rarity scores for sample weighting or bootstrap
        sampling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        ensure_all_finite = "allow-nan" if get_tags(self).input_tags.allow_nan else True
        accept_sparse = "csc" if get_tags(self).input_tags.sparse else "csr"
        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
            accept_sparse=accept_sparse,
            dtype=DTYPE,
            ensure_all_finite=ensure_all_finite,
        )
        check_classification_targets(y)

        rarity_scores = self.calculate_rarity_scores(X, y)

        if self.rarity_adjustment_method == "bootstrap_sampling":
            return self._fit_with_rarity_adjusted_bootstrap_sampling(
                X, y, sample_weight=sample_weight, rarity_scores=rarity_scores
            )
        elif self.rarity_adjustment_method == "sample_weights":
            return self._fit_with_rarity_sample_weights(
                X, y, sample_weight, rarity_scores
            )
        else:
            raise ValueError(
                f"Unknown rarity adjustment method: {self.rarity_adjustment_method}"
            )

    def _fit_with_rarity_sample_weights(
        self, X, y, sample_weight=None, rarity_scores=None
    ):
        """
        Private method to fit the model using rarity scores as sample weights.

        Widely based on the original fit method of sklearn's RandomForestClassifier,
        only modified to include rarity scores for sample weighting.
        """

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight) * np.asarray(rarity_scores)
        else:
            sample_weight = np.asarray(rarity_scores)

        super().fit(X, y, sample_weight)

        return self

    def _fit_with_rarity_adjusted_bootstrap_sampling(
        self, X, y, sample_weight=None, rarity_scores=None
    ):
        """
        Private method used to fit the model using rarity scores
        to adjust bootstrap sampling.

        Widely based on the original fit method of sklearn's RandomForestClassifier,
        only modified to include rarity scores for bootstrap sampling.
        """

        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            ensure_all_finite=False,
        )
        # _compute_missing_values_in_feature_mask checks if X has missing values and
        # will raise an error if the underlying tree base estimator can't handle missing
        # values. Only the criterion is required to determine if the tree supports
        # missing values.
        estimator = type(self.estimator)(criterion=self.criterion)
        missing_values_in_feature_mask = (
            estimator._compute_missing_values_in_feature_mask(
                X, estimator_name=self.__class__.__name__
            )
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

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

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                # This is the only change to the original sklearn .fit method
                # It is needed to pass the rarity_scores to the tree building
                # in order to adjust the bootstrap sampling
                delayed(_parallel_build_trees_rarity_oversampling)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    rarity_scores,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    missing_values_in_feature_mask=missing_values_in_feature_mask,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score and (
            n_more_estimators > 0 or not hasattr(self, "oob_score_")
        ):
            y_type = type_of_target(y)
            if y_type == "unknown" or (
                is_classifier(self) and y_type == "multiclass-multioutput"
            ):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )

            if callable(self.oob_score):
                self._set_oob_score_and_attributes(
                    X, y, scoring_function=self.oob_score
                )
            else:
                self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

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
