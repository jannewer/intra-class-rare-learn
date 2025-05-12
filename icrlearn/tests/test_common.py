"""All scikit-learn common tests.
This makes sure that ICRRandomForestClassifier is compatible with the scikit-learn API.
"""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

from sklearn.utils._test_common.instance_generator import (
    _get_expected_failed_checks,
)
from sklearn.utils.estimator_checks import parametrize_with_checks

from icrlearn import ICRRandomForestClassifier
from icrlearn.utils.discovery import all_estimators


def _get_all_expected_failed_checks(estimator):
    sklearn_expected_failed_checks = _get_expected_failed_checks(estimator)
    icrlearn_expected_failed_checks = {
        ICRRandomForestClassifier: {
            "check_sample_weight_equivalence_on_dense_data": (
                "sample_weight is not equivalent to removing/repeating samples."
            ),
            "check_sample_weight_equivalence_on_sparse_data": (
                "sample_weight is not equivalent to removing/repeating samples."
            ),
        },
    }

    estimator_class = estimator.__class__
    if estimator_class in icrlearn_expected_failed_checks:
        return icrlearn_expected_failed_checks[estimator_class]
    return sklearn_expected_failed_checks


# expected_failed_checks has to be set here to skip tests that are known to fail
# For details see https://github.com/scikit-learn/scikit-learn/issues/16298
# Currently the following tests are disabled in the sklearn test suite:
# - check_sample_weight_equivalence_on_dense_data
# - check_sample_weight_equivalence_on_sparse_data
# (see https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_test_common/instance_generator.py)
# The same tests have to be disabled for the ICRRandomForestClassifier
# as it is a subclass of RandomForestClassifier
@parametrize_with_checks(
    [est() for _, est in all_estimators(type_filter="classifier")],
    expected_failed_checks=_get_all_expected_failed_checks,
)
def test_estimators(estimator, check, request):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
