"""Tests for the intra-class rarity random forest models."""

# Authors: Janne Wernecken
# License: BSD 3 clause

import pytest
from sklearn.datasets import load_iris

from icrlearn import ICRRandomForestClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_icr_lof_rf_classifier(data):
    """Check the internals and behaviour of `ICRRandomForestClassifier`."""
    X, y = data
    clf = ICRRandomForestClassifier()
    assert clf.rarity_measure == "lof"

    clf.fit(X, y)
    assert hasattr(clf, "classes_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
