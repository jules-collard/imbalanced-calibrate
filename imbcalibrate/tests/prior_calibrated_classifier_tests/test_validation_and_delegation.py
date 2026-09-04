import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from imbcalibrate import PriorCalibratedClassifier


class FitParameterClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.received_sample_weight_ = sample_weight
        return self

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


@pytest.fixture
def binary_data():
    return np.zeros((4, 2)), np.array([0, 0, 1, 1])


def test_predict_proba_before_fit_raises():
    with pytest.raises(NotFittedError):
        PriorCalibratedClassifier().predict_proba(np.zeros((1, 2)))


def test_wrong_feature_count_is_rejected(binary_data):
    X, y = binary_data
    clf = PriorCalibratedClassifier(weight=1.0).fit(X, y)

    with pytest.raises(ValueError, match="features"):
        clf.predict_proba(np.zeros((1, 3)))


@pytest.mark.parametrize(
    "y, message",
    [
        (np.array([0, 1, 2, 1]), "Only binary classification"),
        (np.array([0.1, 0.2, 0.3, 0.4]), "Unknown label type"),
    ],
)
def test_non_binary_targets_are_rejected(y, message):
    X = np.zeros((len(y), 2))

    with pytest.raises(ValueError, match=message):
        PriorCalibratedClassifier(weight=1.0).fit(X, y)


def test_non_classifier_estimator_is_rejected(binary_data):
    X, y = binary_data

    with pytest.raises(ValueError, match="must be an instance of"):
        PriorCalibratedClassifier(estimator=DummyRegressor(), weight=1.0).fit(X, y)


def test_fit_parameters_are_forwarded(binary_data):
    X, y = binary_data
    sample_weight = np.array([1.0, 2.0, 3.0, 4.0])
    clf = PriorCalibratedClassifier(estimator=FitParameterClassifier(), weight=1.0).fit(
        X, y, sample_weight=sample_weight
    )

    np.testing.assert_array_equal(clf.estimator_.received_sample_weight_, sample_weight)


def test_original_estimator_is_cloned(binary_data):
    X, y = binary_data
    estimator = LogisticRegression()
    clf = PriorCalibratedClassifier(estimator=estimator, weight=1.0).fit(X, y)

    assert clf.estimator_ is not estimator
    assert not hasattr(estimator, "classes_")


def test_predict_preserves_non_integer_class_labels():
    X = np.zeros((4, 2))
    y = np.array(["negative", "negative", "positive", "positive"])
    clf = PriorCalibratedClassifier(
        estimator=LogisticRegression(random_state=42), weight=1.0
    ).fit(X, y)

    predictions = clf.predict(X)

    assert predictions.dtype.kind in "OUS"
    assert set(predictions).issubset(set(y))
