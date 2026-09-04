.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

Introduction to Imbalanced Learning
===================================

Class imbalance occurs when, for a classification problem, one target variable
class is much less frequent than the other(s). This phenomenon creates challenges when
training classifiers, as models tend to learn more from the majority class, leading to
poor performance on the minority class. With smaller datasets and/or very high class
imbalance, there can also simply be too few samples in the minority class to learn from.
Typical solutions are either data-based or algorithmic.

Data-based techniques involve balancing the distribution of the target variable with
undersampling, oversampling, or a mixture of the two. The ``imbalanced-learn`` library implements
many of these methods whilst integrating with the ``sklearn`` API. See the ``imbalanced-learn``
`user guide`_ for further information on the implemented resampling techniques.

Algorithmic techniques involve training models with respect to a modified loss function, typically
increasing the loss associated with misclassifying the minority class. In particular, the
*weighted cross-entropy loss* is implemented through the ``class_weight`` parameter in ``sklearn``
classifiers, and ``scale_pos_weight`` parameter in the ``xgboost`` and ``lightgbm`` packages.

.. _user guide: https://imbalanced-learn.org/stable/user_guide.html

The Calibration Problem
-----------------------

The main drawback to these methods is that they introduce bias. For both data-based and algorithmic
methods, models are not calibrated to the original data distribution, and instead the new distribution
induced by resampling or weighting the loss function. Resulting probability estimates thus cannot be
interpreted directly, and probability-based evaluation metrics become inaccurate.

Fortunately, in the case of random undersampling, random oversampling, or the weighted cross-entropy
loss, an analytical correction exists which transforms model output back to the true data distribution.
Although this correction is well-documented in the literature, confusion can arise due to differing
notation and parameterisations. Manually applying the correction reduces reproducibility and can create
confusion, for example when saving or sharing a model object. Furthermore, for some parameterisations
(such as ``sklearn``'s ``class_weight``), the correction is a function of both the training data and
parameter value.

To address this issue, this library provides the :class:`PriorCalibratedClassifier` class, which acts
as a calibration wrapper (similar to :class:`sklearn.calibration.CalibratedClassifierCV`) around any
``sklearn``-compatible classifier. In particular, the implementation automatically calculates the
correct correction by inspecting the sub-estimator's (or ``imbalanced-learn`` resampler's) parameters.

imbalanced-calibrate API
========================

:class:`PriorCalibratedClassifier` is fully integrated within the ``sklearn`` API, acting as as Meta-Estimator
and Classifier object. As with all ``sklearn`` estimators, it implements the ``fit``, ``predict_proba`` and
``predict`` methods.

The :class:`PriorCalibratedClassifier` takes an (untrained) classifier and optional ``weight``
parameter at instantiation, for example::

    >>> from imbcalibrate import PriorCalibratedClassifier
    >>> from sklearn.linear_model import LogisticRegression

    >>> clf = PriorCalibratedClassifier(LogisticRegression(class_weight='balanced'))

Calling ``clf.fit(X, y)`` fits the sub-estimator to the data, and infers the weight from the sub-estimator's
parameters (and training data when applicable) when possible. If ``weight`` is provided at instatiation,
it overrides any inferred weight. Currently weight inference is implemented for the following estimator objects
(in order of priority):

* :class:`imblearn.pipeline.Pipeline`: weight is inferred from the first instance of :class:`RandomOverSampler` or :class:`RandomUnderSampler` encountered in the pipeline steps.
* :class:`sklearn.pipeline.Pipeline`: weight is inferred from the last step of the pipeline, provided it is as classifier (also applies to :class:`imblearn.pipeline.Pipeline` if no sampler is found).
* Any ``sklearn`` classifier implementing the ``class_weight`` parameter.
* :class:`xgboost.XGBClassifier`, :class:`lightgbm.LGBClassifier`, or any similar estimator which implements the ``scale_pos_weight`` parameter.

Once fitted, calls to ``clf.predict_proba(X)`` and ``clf.predict(X)`` use the calibration-corrected probability
estimates. Crucially, this behaviour persists with model object saving/loading, as the weight is set and saved at
``fit`` time. The fitted sub-estimator can be accessed through the ``estimator_`` attribute.

Mathematical Formulation
========================
