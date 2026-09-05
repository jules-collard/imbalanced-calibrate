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
``sklearn``-compatible binary classifier. In particular, the implementation automatically calculates the
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

Weighted Cross-Entropy Loss
---------------------------

Let :math:`Y \sim \mathbb{P}` be a binary random variable taking values 0 and 1, with :math:`\pi_0 = \mathbb{P}(Y=0)`
and :math:`\pi_1 = \mathbb{P}(Y=1)`. Recall the binary cross-entropy loss is given by

.. math::

    L(y, \hat{p}) = -y\log{\hat{p}} - (1-y)\log{(1-\hat{p})},

for a probability estimate :math:`\hat{p}`. W.l.o.g., we assume that :math:`Y=1` is the minority class, in which case
the *weighted* cross-entropy loss can be defined as

.. math::
    
    L^w(y, \hat{p}) = -y w \log{\hat{p}} - (1-y)\log{(1-\hat{p})},

for some weight :math:`w > 1`. Now, we have

.. math::

    \mathbb{E}_\mathbb{P}[L^w(y, \hat{p})] &= \mathbb{P}(Y=1)\mathbb{E}[L^w(1, \hat{p})|Y=1] +
    \mathbb{P}(Y=0)\mathbb{E}[L^w(0, \hat{p})|Y=0] \\
    &= w \pi_1 \mathbb{E}[L(1, \hat{p})|Y=1] + \pi_0 \mathbb{E}[L(0, \hat{p})|Y=0] \\
    &\propto \mathbb{P}_w(Y=1)\mathbb{E}[L(1, \hat{p})|Y=1] +
    \mathbb{P}_w(Y=0)\mathbb{E}[L(0, \hat{p})|Y=0] \\
    &= \mathbb{E}_{\mathbb{P}_w}[L(y,\hat{p})],

where :math:`\mathbb{P}_w` is an implicit probability measure induced by :math:`w`, with
:math:`\mathbb{P}_w(Y=0)=\pi_0/z` and :math:`\mathbb{P}_w(Y=1)=(w\pi_1)/z`, where :math:`z = \pi_0 + w\pi_1`
is a normalising constant. We assume that :math:`\mathbb{P}(\cdot|Y) = \mathbb{P}_w(\cdot|Y)`, since the
weight is entirely determined by the label :math:`Y`.

Notice that the prior odds induced by the weighting are given by

.. math::
    O_w = \frac{w \pi_1 / z}{\pi_0 / z} = w \frac{\pi_1}{\pi_0} = w \cdot O.

That is, minimising a weighted loss (with weight :math:`w`) under the true distribution :math:`\mathbb{P}`
is proportional to minimising an unweighted loss under the *artificial data distribution* :math:`\mathbb{P}_w`,
for which the odds of the positive class are multiplied by :math:`w`.

Resampling Methods
------------------

In fact, there is a one-to-one relationship between the changes in prior odds induced by the weighted loss
and resampling methods. The methods below rely on the assumption that the conditional distributions are not
affected by resampling, i.e. :math:`P(\cdot|Y, S) = P(\cdot|Y)`, where :math:`S` is a random variable indicating
whether an observation is included in the resampled dataset. In practice, this assumption only holds for
`random oversampling`_ and `random undersampling`_ .

.. _random undersampling: https://imbalanced-learn.org/stable/under_sampling.html#random-under-sampling
.. _random oversampling: https://imbalanced-learn.org/stable/over_sampling.html#naive-random-over-sampling

Oversampling
^^^^^^^^^^^^

Let :math:`N_0` and :math:`N_1` be the number of samples of each class in the dataset, so that the empirical
probability distribution is given by :math:`\hat{\mathbb{P}}(Y=1) = N_1/(N_0+N_1)`, with empirical odds ratio
:math:`\hat{O}=N_1/N_0`. Suppose we oversample with rate :math:`w > 1`, so that the oversampled dataset has
:math:`wN_1` samples of the positive/minority class. The new empirical odds ratio is given by

.. math::
    \hat{O}_w = \frac{w N_1}{N_0} = w \hat{O}.

That is, oversampling with rate :math:`w` results in the same shift in prior odds ratio as using the weight :math:`w`
in a weighted cross-entropy loss.

Undersampling
^^^^^^^^^^^^^

Similarly, suppose we undersample with rate :math:`1/w`, so that the undersampled dataset has :math:`N_0/w` samples
of the negative/majority class. We obtain the same shift in odds ratio:

.. math::
    \hat{O}_w = \frac{N_1}{N_0/w} = w \hat{O}.

Imbalanced-Learn Parameterisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``imbalanced-learn`` samplers, the sampling rates are specified by the ``sampling_strategy`` parameter, which is defined
as "the desired ratio of the number of samples in the minority class over the number of samples in the majority class
after resampling." That is, the ``sampling_strategy`` parameter corresponds to the desired odds ratio :math:`\hat{O}_w`.
To recover the weight, we simply have

.. math::
    w = \hat{O}_w / \hat{O} = \frac{N_1^s}{N_0^s} \cdot \frac{N_0}{N_1},

where :math:`N_0^s` and :math:`N_1^s` are the number of samples in the negative and positive classes respectively in the
resampled dataset.

Applying the Correction
-----------------------

We have established the equivalence in odds ratio shift between the weighted loss, over- and under-sampling in terms
of a weight `w`. It remains to transform the estimated (uncalibrated) probability estimates :math:`p_w` to the true
data distribution.

.. math::
    O_w &= w \cdot O \\
    \frac{p_w}{1-p_w} &= w \cdot \frac{p}{1-p} \\
    &\vdots \\
    p &= \frac{\frac{p_w}{w(1-p_w)}}{1 + \frac{p_w}{w(1-p_w)}} \\
    &= \frac{p_w}{w(1-p_w) + p_w}