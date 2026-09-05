.. _quick_start:

###############
Getting started
###############

This package provides ``sklearn``-compatible calibration methods which correct model output for the bias induced
by certain imbalanced learning techniques. It supports class weighting in ``sklearn`` binary classifiers 
(including `XGBoost`_  and `LightGBM`_ ), and resampling methods from the `imbalanced-learn`_ package.

.. _XGBoost: https://xgboost.readthedocs.io/en/stable/
.. _LightGBM: https://lightgbm.readthedocs.io/en/stable/
.. _imbalanced-learn: https://imbalanced-learn.org/stable/

Prerequisites
=============

``imbalanced-calibrate`` requires the following dependencies:

* Python (>=3.10)
* NumPy (>=2.0.2)
* Scikit-learn (>=1.6.0)

Additionally, ``imbalanced-calibrate`` requires the following optional dependencies:

* Imbalanced-learn (>=0.14.2), for calibration after using ``imbalanced-learn`` resampling methods.

Install
=======

``imbalanced-calibrate`` is currently available on the PyPI repository and you can install it via ``pip``: ::
    pip install imbalanced-calibrate

Or with ``uv``: ::
    uv add imbalanced-calibrate

The optional dependencies for resampling methods can be installed using ``pip install imbalanced-calibrate[resampling]`` or ``uv add "imbalanced-calibrate[resampling]"``.

Contribute
==========

You can contribute to this package through a Pull Request on `GitHub`_, subject to appropriate unit testing and review.

.. _GitHub: https://github.com/jules-collard/imbalanced-calibrate