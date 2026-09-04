.. _quick_start:

###############
Getting started
###############

This package provides analytical prior calibration utilities to correct probability estimates from binary classifiers trained with class rebalancing techniques.

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

From PyPI
---------

``imbalanced-calibrate`` is currently available on the PyPI repository and you can install it via ``pip``: ::
    pip install imbalanced-calibrate

Or with ``uv``: ::
    uv add imbalanced-calibrate

The optional dependencies for resampling methods can be installed using ``pip install imbalanced-calibrate[resampling]`` or ``uv add "imbalanced-calibrate[resampling]".

Contribute
==========

You can contribute to this package through a Pull Request on `GitHub`_, subject to appropriate unit testing and review.

.. _GitHub: https://github.com/jules-collard/imbalanced-calibrate