.. _quick_start:

###############
Getting started
###############

This package serves as a skeleton package aiding at developing compatible
scikit-learn contribution.

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

From PyPi
---------

``imbalanced-calibrate`` is currently available on the PyPi’s repositories and you can install it via ``pip``: ::
    pip install imbalanced-calibrate

Or with ``uv``: ::
    uv add imbalanced-calibrate

The optional dependencies for resampling methods can be installed using ``pip install imbalanced-calibrate[resampling]`` or ``uv add "imbalanced-calibrate[resampling]".

Contribute
==========

You can contribute to this code through a Pull Request on `GitHub`_. Please, make sure that your code is coming with unit tests to ensure full coverage and continuous integration in the API.

.. _GitHub: https://github.com/jules-collard/imbalanced-calibrate