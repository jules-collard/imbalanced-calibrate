import warnings

import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from imbcalibrate import PriorCalibratedClassifier


@pytest.fixture
def dummy_data():
    X = np.random.rand(20, 5)
    y = np.array([0]*15 + [1]*5)
    return X, y

def test_explicit_weight_overrides_estimator(dummy_data):
    X, y = dummy_data
    # The estimator sets weight to 5.0
    base_clf = XGBClassifier(scale_pos_weight=5.0)
    # But the explicit parameter is 2.0
    clf = PriorCalibratedClassifier(estimator=base_clf, weight=2.0)
    clf.fit(X, y)

    # Explicit weight should win
    assert clf.weight_ == pytest.approx(2.0)

def test_explicit_weight_overrides_pipeline_sampler(dummy_data):
    X, y = dummy_data
    pipe = ImblearnPipeline([
        ('rus', RandomUnderSampler(sampling_strategy="majority", random_state=42)),
        ('clf', LogisticRegression())
    ])
    # Sampler would normally dictate a weight change, but explicit weight should win
    clf = PriorCalibratedClassifier(estimator=pipe, weight=10.0)
    clf.fit(X, y)

    assert clf.weight_ == pytest.approx(10.0)

def test_missing_weight_issues_warning(dummy_data):
    X, y = dummy_data
    # GaussianNB does not have class_weight or scale_pos_weight
    base_clf = GaussianNB()
    clf = PriorCalibratedClassifier(estimator=base_clf, weight=None)

    with pytest.warns(UserWarning, match="Defaulting to weight=1.0."):
        clf.fit(X, y)

    assert clf.weight_ == 1.0

def test_default_estimator_is_logistic_regression(dummy_data):
    X, y = dummy_data
    clf = PriorCalibratedClassifier(estimator=None, weight=2.0)
    clf.fit(X, y)

    # Should instantiate a LogisticRegression internally
    assert isinstance(clf.estimator_, LogisticRegression)


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_non_positive_weight_is_rejected(dummy_data, weight):
    X, y = dummy_data
    clf = PriorCalibratedClassifier(weight=weight)

    with pytest.raises(ValueError, match="must be a positive float"):
        clf.fit(X, y)


def test_class_weight_none_infers_one_without_warning(dummy_data):
    X, y = dummy_data
    clf = PriorCalibratedClassifier(
        estimator=RandomForestClassifier(class_weight=None, random_state=42)
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        clf.fit(X, y)

    assert clf.weight_ == pytest.approx(1.0)
    assert len(recorded) == 0


def test_scale_pos_weight_takes_precedence_over_class_weight(dummy_data):
    X, y = dummy_data
    estimator = XGBClassifier(
        scale_pos_weight=3.0,
        class_weight={0: 1.0, 1: 7.0},
        random_state=42,
    )
    clf = PriorCalibratedClassifier(estimator=estimator)
    clf.fit(X, y)

    assert clf.weight_ == pytest.approx(3.0)
