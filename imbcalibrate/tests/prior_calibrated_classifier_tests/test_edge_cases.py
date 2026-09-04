import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
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
