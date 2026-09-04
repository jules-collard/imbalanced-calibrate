import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from imbcalibrate import PriorCalibratedClassifier


class MockClassifier(ClassifierMixin, BaseEstimator):
    """A mock classifier that returns fixed probabilities for testing math."""

    def __init__(self, probas):
        self.probas = probas
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        # Return the same fixed probas for however many rows are in X
        return np.tile(np.array(self.probas), (len(X), 1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@pytest.fixture
def dummy_X_y():
    return np.zeros((2, 5)), np.array([0, 1])


def test_calibration_formula_exact_values(dummy_X_y):
    X, y = dummy_X_y
    # Uncalibrated probabilities: P(y=1) = 0.8, P(y=0) = 0.2
    base_probas = [0.2, 0.8]
    mock_clf = MockClassifier(probas=base_probas)

    # Weight w = 4.0 means the positive class was oversampled/weighted by 4x.
    # The calibration should reduce the predicted probability of the positive class.
    # Formula: P_calib = P_uncalib / (P_uncalib + (1 - P_uncalib) * w)
    # P_calib(y=1) = 0.8 / (0.8 + 0.2 * 4.0) = 0.8 / 1.6 = 0.5
    clf = PriorCalibratedClassifier(estimator=mock_clf, weight=4.0)
    clf.fit(X, y)

    probas = clf.predict_proba(X)
    expected_probas = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(probas, expected_probas)


def test_weight_of_one_is_identity(dummy_X_y):
    X, y = dummy_X_y
    base_probas = [0.3, 0.7]
    mock_clf = MockClassifier(probas=base_probas)

    clf = PriorCalibratedClassifier(estimator=mock_clf, weight=1.0)
    clf.fit(X, y)

    probas = clf.predict_proba(X)
    expected_probas = np.array([base_probas, base_probas])
    np.testing.assert_allclose(probas, expected_probas)


def test_predict_threshold_shift(dummy_X_y):
    X, y = dummy_X_y
    # P(y=1) = 0.6. Normally, predict() would return 1 (since 0.6 >= 0.5)
    base_probas = [0.4, 0.6]
    mock_clf = MockClassifier(probas=base_probas)

    # Weight w = 3.0.
    # Calibrated P(y=1) = 0.6 / (0.6 + 0.4 * 3.0) = 0.6 / 1.8 = 0.333
    # Calibrated P(y=1) < 0.5, so calibrated predict() should return 0.
    clf = PriorCalibratedClassifier(estimator=mock_clf, weight=3.0)
    clf.fit(X, y)

    preds = clf.predict(X)
    np.testing.assert_array_equal(preds, np.array([0, 0]))


@pytest.mark.parametrize("positive_probability", [0.0, 1.0])
def test_calibration_handles_probability_boundaries(dummy_X_y, positive_probability):
    X, y = dummy_X_y
    mock_clf = MockClassifier(probas=[1.0 - positive_probability, positive_probability])
    clf = PriorCalibratedClassifier(estimator=mock_clf, weight=4.0)
    clf.fit(X, y)

    probas = clf.predict_proba(X)

    assert np.all(np.isfinite(probas))
    np.testing.assert_allclose(probas.sum(axis=1), 1.0)
    np.testing.assert_allclose(probas[:, 1], np.full(len(X), positive_probability))
