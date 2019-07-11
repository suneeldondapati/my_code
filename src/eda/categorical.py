#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Filename: eda
Date: 2019-05-09 11:28
Project: AXA
AUTHOR: Suneel Dondapati
"""


import numpy as np
import pandas as pd
import scipy.stats as ss


__all__ = [
    'cramers_corrected_stat',
]


def cramers_corrected_stat(confusion_matrix: pd.DataFrame):
    """Calculate bias corrected cramer's V

    Arguments:
    ----------
    confusion_matrix: pd.DataFrame
        Contingency table of two nominal variables

    Returns:
    --------
    cramers_v: float
        Cramer's V value for the input nominal variables

    Examples:
    --------
    >>> cramers_corrected_stat(pd.crosstab(df['More customer service calls'], df['Churn_Flag']))
    0.3136

    References:
    -----------
    Cramer's V is the measure of association between two
    nominal variables, giving a value between 0 and +1
    '0' corresponds to no association
    '1' corresponds to complete association, value can reach 1
    only when two variables are equal to each other.

    [1] https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1) / (n - 1)))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return cramers_v
