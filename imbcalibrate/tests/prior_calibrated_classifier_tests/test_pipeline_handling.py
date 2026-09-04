import numpy as np
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from imbcalibrate import PriorCalibratedClassifier


@pytest.fixture
def imbalanced_data():
    X = np.random.rand(100, 5)
    y = np.array([0]*80 + [1]*20)
    return X, y

def test_sklearn_pipeline_inference(imbalanced_data):
    X, y = imbalanced_data
    pipe = SklearnPipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(scale_pos_weight=2.0))
    ])
    clf = PriorCalibratedClassifier(estimator=pipe, weight=None)
    clf.fit(X, y)

    assert clf.weight_ == pytest.approx(2.0)

def test_imblearn_pipeline_random_under_sampler(imbalanced_data):
    X, y = imbalanced_data
    # 80 zeros, 20 ones.
    # sampling_strategy=0.5 means N_1 / N_0 = 0.5 -> N_0 = N_1 / 0.5 = 20 / 0.5 = 40.
    # The negative class is undersampled from 80 to 40. The positive class is untouched.
    # Effective weight on the positive class relative to the original distribution:
    # Original ratio N_1/N_0 = 20/80 = 0.25
    # Sampled ratio N_1/N_0 = 20/40 = 0.5
    # Weight = 0.5 / 0.25 = 2.0
    pipe = ImblearnPipeline([
        ('rus', RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
        ('clf', LogisticRegression())
    ])
    clf = PriorCalibratedClassifier(estimator=pipe, weight=None)
    clf.fit(X, y)

    assert clf.weight_ == pytest.approx(2.0)

def test_imblearn_pipeline_random_over_sampler(imbalanced_data):
    X, y = imbalanced_data
    # sampling_strategy="minority" means N_1 will be oversampled to match N_0 (80).
    # Original N_1/N_0 = 20/80 = 0.25
    # New N_1/N_0 = 80/80 = 1.0
    # Weight = 1.0 / 0.25 = 4.0
    pipe = ImblearnPipeline([
        ('ros', RandomOverSampler(sampling_strategy="minority", random_state=42)),
        ('clf', LogisticRegression())
    ])
    clf = PriorCalibratedClassifier(estimator=pipe, weight=None)
    clf.fit(X, y)

    assert clf.weight_ == pytest.approx(4.0)


def test_imblearn_pipeline_random_over_sampler_negative_class(imbalanced_data):
    X, y = imbalanced_data
    # Explicitly oversample class 0 from 80 to 160 samples.
    pipe = ImblearnPipeline([
        ("ros", RandomOverSampler(sampling_strategy={0: 160}, random_state=42)),
        ("clf", LogisticRegression()),
    ])
    clf = PriorCalibratedClassifier(estimator=pipe, weight=None)
    clf.fit(X, y)

    # Original ratio N_0/N_1 = 80/20; resampled ratio N_0/N_1 = 160/20.
    assert clf.weight_ == pytest.approx(0.5)

def test_unsupported_imblearn_resampler(imbalanced_data):
    X, y = imbalanced_data
    # SMOTE is not RandomOverSampler or RandomUnderSampler
    pipe = ImblearnPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression())
    ])
    clf = PriorCalibratedClassifier(estimator=pipe, weight=None)

    with pytest.warns(
        UserWarning,
        match="Imbalanced-learn pipeline detected but sampler not found/supported."
    ):
        clf.fit(X, y)

    assert clf.weight_ == 1.0
