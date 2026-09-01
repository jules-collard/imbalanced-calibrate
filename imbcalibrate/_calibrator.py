from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class PriorCalibratedClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

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
    >>> from skltemplate import TemplateClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = TemplateClassifier().fit(X, y)
    >>> clf.predict(X)
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
        "e": [str],
    }

    def __init__(self, estimator: None | ClassifierMixin=None, weight: None | float=None):
        self.estimator = estimator
        self.weight = weight

    def fit(self, X, y, **fit_params):
        """
        Clones the base estimator, fits it to the training data, and sets attributes.
        """
        X, y = validate_data(
            self, X=X, y=y, reset=True,
            estimator="PriorCalibratedClassifier"
        )
        check_classification_targets(y)

        if self.estimator is None:
            self.estimator_ = DummyClassifier(strategy="uniform")
        else:
            self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = getattr(self.estimator_, "classes_")

        if hasattr(self.estimator_, "scale_pos_weight"): # XGBoost and LightGBM
            if self.weight is not None:
                warn("The provided weight will be ignored since the estimator has 'scale_pos_weight' attribute.")
            self.weight_ = getattr(self.estimator_, "scale_pos_weight", 1.0)
        elif hasattr(self.estimator_, "class_weight_"): # Sklearn Logistic Regression
            if self.weight is not None:
                warn("The provided weight will be ignored since the estimator has 'class_weight_' attribute.")
            cw = getattr(self.estimator_, "class_weight_")
            if isinstance(cw, dict):
                # Calculate ratio: weight of class 1 / weight of class 0
                weight_0 = cw.get(self.classes_[0], 1.0)
                weight_1 = cw.get(self.classes_[1], 1.0)
                self.weight_ = weight_1 / weight_0 if weight_0 != 0 else 1.0
            elif cw == 'balanced':
                # Calculate ratio based on actual sample counts in 'y'
                n_pos = np.sum(y == self.classes_[1])
                n_neg = np.sum(y == self.classes_[0])
                self.weight_ = n_neg / n_pos if n_pos > 0 else 1.0
        else:
            self.weight_ = self.weight if self.weight is not None else 1.0
            if self.weight is None:
                warn("No weight provided and estimator does not have 'scale_pos_weight' or 'class_weight_'. Defaulting to weight=1.0.")

        self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        """
        Extracts raw probabilities and applies the mathematical prior correction.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, estimator="PriorCalibratedClassifier")

        raw_probs = self.estimator_.predict_proba(X)
        p_raw = raw_probs[:, 1]

        p_calibrated = p_raw / (p_raw + self.weight_ * (1 - p_raw))

        return np.vstack([1 - p_calibrated, p_calibrated]).T

    def predict(self, X):
        """
        Converts calibrated probabilities back into discrete class predictions (0 or 1).
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False, estimator="PriorCalibratedClassifier")

        calibrated_probs = self.predict_proba(X)
        return self.classes_[np.argmax(calibrated_probs, axis=1)]
