:notoc:

#############################################
imbalanced-calibrate documentation
#############################################

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/jules-collard/imbalanced-calibrate>`__ |
`Issues & Ideas <https://github.com/jules-collard/imbalanced-calibrate/issues>`__ |

Imbalanced-calibrate (imported as `imbcalibrate`) provides `scikit-learn`_ -compatible calibration methods
which correct probability estimates for the bias introduced by imbalanced learning techniques.

.. _scikit-learn: https://scikit-learn.org/stable/

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting started
        :img-top: _static/img/index_getting_started.svg
        :class-card: intro-card
        :shadow: md

        Dependencies, installation instructions & contribution guide.

        +++

        .. button-ref:: quick_start
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the getting started guideline

    .. grid-item-card::  User guide
        :img-top: _static/img/index_user_guide.svg
        :class-card: intro-card
        :shadow: md

        The user guide explains why imbalanced learning methods require calibration, and how to
        easily apply these methods using `imbalanced-calibrate`.

        +++

        .. button-ref:: user_guide
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the user guide

    .. grid-item-card::  API reference
        :img-top: _static/img/index_api.svg
        :class-card: intro-card
        :shadow: md

        Detailed description of the `imbalanced-calibrate` API.

        +++

        .. button-ref:: api
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide

    .. grid-item-card::  Examples
        :img-top: _static/img/index_examples.svg
        :class-card: intro-card
        :shadow: md

        A set of examples.

        +++

        .. button-ref:: general_examples
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the gallery of examples


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    quick_start
    user_guide
    api
    auto_examples/index
