import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from imbcalibrate import PriorCalibratedClassifier


@pytest.fixture
def imbalanced_data():
    # 90 zeros, 10 ones
    X = np.random.rand(100, 5)
    y = np.array([0]*90 + [1]*10)
    return X, y

def test_infer_sklearn_class_weight_dict(imbalanced_data):
    X, y = imbalanced_data
    clf = PriorCalibratedClassifier(
        estimator=RandomForestClassifier(class_weight={0: 1, 1: 5}, random_state=42),
        weight=None
    )
    clf.fit(X, y)

    # Check that weight is correctly inferred as 5.0 / 1.0 = 5.0
    assert clf.weight_ == pytest.approx(5.0)


@pytest.mark.parametrize(
    "class_weight, expected_weight",
    [({1: 5}, 5.0), ({0: 2}, 0.5), ({0: 0, 1: 5}, 1.0)],
)
def test_infer_class_weight_dict_defaults_and_zero_denominator(
    imbalanced_data, class_weight, expected_weight
):
    X, y = imbalanced_data
    clf = PriorCalibratedClassifier(
        estimator=RandomForestClassifier(
            class_weight=class_weight, random_state=42
        )
    )
    clf.fit(X, y)

    assert clf.weight_ == pytest.approx(expected_weight)

def test_infer_sklearn_class_weight_balanced(imbalanced_data):
    X, y = imbalanced_data
    clf = PriorCalibratedClassifier(
        estimator=LogisticRegression(class_weight="balanced", random_state=42),
        weight=None
    )
    clf.fit(X, y)

    # In balanced mode: weight for class i = n_samples / (n_classes * n_samples_i)
    # w_0 = 100 / (2 * 90) = 100/180 = 0.555...
    # w_1 = 100 / (2 * 10) = 100/20 = 5.0
    # Effective relative weight = w_1 / w_0 = 5.0 / (100/180) = 9.0
    assert clf.weight_ == pytest.approx(9.0)

def test_infer_xgboost_scale_pos_weight(imbalanced_data):
    X, y = imbalanced_data
    clf = PriorCalibratedClassifier(
        estimator=XGBClassifier(scale_pos_weight=3.5, random_state=42),
        weight=None
    )
    clf.fit(X, y)

    assert clf.weight_ == pytest.approx(3.5)

def test_infer_lightgbm_scale_pos_weight(imbalanced_data):
    X, y = imbalanced_data
    clf = PriorCalibratedClassifier(
        estimator=LGBMClassifier(scale_pos_weight=4.0, random_state=42),
        weight=None
    )
    clf.fit(X, y)

    assert clf.weight_ == pytest.approx(4.0)

def test_infer_lightgbm_class_weight_balanced(imbalanced_data):
    X, y = imbalanced_data
    clf = PriorCalibratedClassifier(
        estimator=LGBMClassifier(class_weight="balanced", random_state=42),
        weight=None
    )
    clf.fit(X, y)

    # Should resolve to 9.0 similarly to sklearn LogisticRegression
    assert clf.weight_ == pytest.approx(9.0)
