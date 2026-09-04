from warnings import warn

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    _IMBLEARN_INSTALLED = True
except ImportError:
    _IMBLEARN_INSTALLED = False


class PriorCalibratedClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """A meta-estimator which analytically calibrates the output of a binary classifier
    to account for class balancing techniques.

    Parameters
    ----------
    estimator : estimator instance, default=None
        The classifier whose output need to be calibrated to provide more accurate
        `predict_proba` outputs. If estimator is a `Pipeline`, the last step of the
        pipeline must be a classifier. If estimator is an imbalanced-learn `Pipeline`,
        only `RandomUnderSampler` or `RandomOverSampler` are recognised. If
        estimator is None, a default `LogisticRegression` classifier will be used.

    weight : float, default=None
        The weight to be used for the prior calibration. If None, the weight will be
        inferred from the estimator's `scale_pos_weight` or `class_weight` attributes if
        available, or the resampler's sampling strategy when applicable.
        When provided, weight acts as an override and will be used instead of
        the estimator's attributes. If the estimator does not have these attributes and
        weight is `None`, a warning is issued and the weight will default to 1.0.

    Attributes
    ----------
    estimator_ : estimator instance
        The fitted (uncalibrated) estimator.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from imbcalibrate import PriorCalibratedClassifier
    >>>
    >>> X, y = make_classification(
    ...     n_samples=1000, weights=[0.9, 0.1], random_state=0
    ... )
    >>> classifier = PriorCalibratedClassifier(
    ...     estimator=LogisticRegression(class_weight="balanced", random_state=0)
    ... )
    >>> classifier.fit(X, y) # doctest: +ELLIPSIS
    PriorCalibratedClassifier(...)
    >>> probabilities = classifier.predict_proba(X[:5])
    >>> probabilities.shape
    (5, 2)
    """

    # For @_fit_context decorator
    _parameter_constraints = {
        "estimator": [ClassifierMixin, Pipeline, None],
        "weight": [float, None],
    }

    def __init__(
        self,
        estimator: ClassifierMixin | Pipeline | None = None,
        weight: float | None = None,
    ):
        self.estimator = estimator
        self.weight = weight

    def __sklearn_tags__(self):
        """Define estimator tags for scikit-learn >= 1.6."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and `n_features`
            is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        **fit_params : kwargs
            Additional fit parameters to pass to the underlying estimator's `fit`
            method.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Data validation and type checking
        X, y = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
        )
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {y_type}."
            )

        # Estimator validation
        est = self.estimator if self.estimator is not None else LogisticRegression()
        if not is_classifier(est):
            raise ValueError(
                "The base estimator should be a classifier. "
                f"Passed {type(est).__name__} instead."
            )

        self.estimator_ = clone(est)
        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = self.estimator_.classes_

        # ==========================================
        # Weight Resolution Logic
        # ==========================================

        sampler = None
        final_est = None
        inferred_weight = None

        # Extract classifier and/or sampler from pipeline if applicable
        if isinstance(self.estimator_, Pipeline):
            final_est = self.estimator_.steps[-1][1]
            if _IMBLEARN_INSTALLED and isinstance(self.estimator_, ImbPipeline):
                for _, step in self.estimator_.steps:
                    if isinstance(step, (RandomUnderSampler, RandomOverSampler)):
                        sampler: RandomUnderSampler | RandomOverSampler = step
                        break
                if sampler is None:
                    warn(
                        "Imbalanced-learn pipeline detected but sampler not "
                        "found/supported."
                    )
        else:
            final_est = self.estimator_

        if sampler is not None:
            # Original counts
            n_pos = np.sum(y == self.classes_[1])
            n_neg = np.sum(y == self.classes_[0])

            # Fallback to original count if class wasn't touched by sampler
            resampled_counts: dict = sampler.sampling_strategy_
            n_pos_resampled = resampled_counts.get(self.classes_[1], n_pos)
            # Oversampling strategy dict gives number of additional samples, not total
            if isinstance(sampler, RandomOverSampler):
                n_pos_resampled += n_pos
            n_neg_resampled = resampled_counts.get(self.classes_[0], n_neg)

            if any(
                [n_pos_resampled == 0, n_neg_resampled == 0, n_pos == 0, n_neg == 0]
            ):
                raise ValueError("Zero samples encountered for a class.")

            inferred_weight = (n_neg / n_pos) * (n_pos_resampled / n_neg_resampled)

        elif hasattr(final_est, "scale_pos_weight"):  # XGBoost and LightGBM
            inferred_weight = final_est.scale_pos_weight

        elif hasattr(final_est, "class_weight"):  # Sklearn classifiers
            cw = final_est.class_weight
            if cw is None:
                inferred_weight = 1.0
            elif isinstance(cw, dict):
                # Calculate ratio: weight of class 1 / weight of class 0
                weight_0 = cw.get(self.classes_[0], 1.0)
                weight_1 = cw.get(self.classes_[1], 1.0)
                inferred_weight = weight_1 / weight_0 if weight_0 != 0 else 1.0
            elif cw == "balanced":
                # Calculate ratio based on actual sample counts in 'y'
                n_pos = np.sum(y == self.classes_[1])
                n_neg = np.sum(y == self.classes_[0])
                inferred_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # ==========================================
        # Overrides and Warnings
        # ==========================================

        if self.weight is not None:  # override with provided weight
            if self.weight <= 0:
                raise ValueError(
                    f"The weight must be a positive float. Got weight={self.weight}."
                )
            if inferred_weight is not None:
                warn(
                    f"Weight parameter override: Using provided weight={self.weight} "
                    f"instead of inferred weight={inferred_weight}."
                )
            self.weight_ = self.weight
        else:  # use inferred weight if available, else default to 1.0
            if inferred_weight is not None:
                self.weight_ = inferred_weight
            else:
                warn(
                    "No weight provided and could not infer weight from the estimator. "
                    "Defaulting to weight=1.0."
                )
                self.weight_ = 1.0

        self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        """
        Calibrated probabilities of classification.

        This function returns calibrated probabilities of classification according to
        each class on an array of test vectors `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict_proba`.

        Returns
        -------
        C : ndarray, shape (n_samples, n_classes)
            The array of calibrated probabilities of classification according to each
            class.
        """
        p_calibrated = self._calibrate_proba(X)

        return np.vstack([1 - p_calibrated, p_calibrated]).T

    def predict(self, X):
        """Predict the target of new samples.

        The predicted class is the class that has the highest probability, and can thus
        be different from the prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples, as accepted by `estimator.predict_proba`.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted class.
        """
        calibrated_probs = self.predict_proba(X)
        return self.classes_[np.argmax(calibrated_probs, axis=1)]

    def _calibrate_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        raw_probs = self.estimator_.predict_proba(X)
        p_raw = raw_probs[:, 1]
        p_calibrated = p_raw / (p_raw + self.weight_ * (1 - p_raw))
        return p_calibrated
