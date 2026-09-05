# imbalanced-calibrate - Analytical Calibration for Imbalanced Learning

![tests](https://github.com/jules-collard/imbalanced-calibrate/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/jules-collard/imbalanced-calibrate/graph/badge.svg?token=2VRRMZ3LE1)](https://codecov.io/gh/jules-collard/imbalanced-calibrate)
![doc](https://github.com/jules-collard/imbalanced-calibrate/actions/workflows/deploy-gh-pages.yml/badge.svg)

This package provides [sklearn](https://scikit-learn.org/stable/)-compatible calibration methods which correct model output for the bias induced by certain imbalanced learning techniques. It supports class weighting in `sklearn` binary classifiers (including [XGBoost](https://xgboost.readthedocs.io/en/stable/) and [LightGBM](https://lightgbm.readthedocs.io/en/stable/)), and resampling methods from the [imbalanced-learn](https://imbalanced-learn.org/stable/) package.

## Documentation

Installation instructions, usage examples and mathematical justification can be found in the [documentation](https://jules-collard.github.io/imbalanced-calibrate/).

## Prerequisites

`imbalanced-calibrate` requires the following dependencies:

- Python (>=3.10)
- NumPy (>=2.0.2)
- Scikit-learn (>=1.6.0)

Additionally, `imbalanced-calibrate` requires the following optional dependencies:

- Imbalanced-learn (>=0.14.2), for calibration after using ``imbalanced-learn`` resampling methods.

## Install

`imbalanced-calibrate` is currently available on the PyPI repository and you can install it via `pip`:

```
pip install imbalanced-calibrate
```

Or with `uv`:

```
uv add imbalanced-calibrate
```

The optional dependencies for resampling methods can be installed using `pip install imbalanced-calibrate[resampling]` or `uv add "imbalanced-calibrate[resampling]"`.