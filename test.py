#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Filename: test.py
Date: 2019-08-30 16:34
Project: my_code
AUTHOR: Suneel Dondapati
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from myml.model.classifier import XGBClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
y_mod = np.where(y == 1, 1, 0)
X_df = pd.DataFrame(X, columns=iris.feature_names)
y_df = pd.Series(y_mod, name='Species')
print(y_df.value_counts())
xgbc = XGBClassifier()
xgbc.fit(X_df, y_df)
print(xgbc.feature_importance_)
xgbc_metric = xgbc.classification_metric(X_df, y_df, labels={1: 'Virginica', 0: 'Non-Virginica'})
print(f'AUC: {xgbc_metric.auc_}')
for threshold in [0.3, 0.5, 0.8]:
    print('Metrics At threshold: {threshold}')
    cfm_threshold = xgbc_metric.confusion_matrix(threshold)
    print(f'Fancy Confusion Matrix: \n{xgbc_metric.cfm}')
    print(f'Confusion Matrix table: \n{cfm_threshold.table_}')
    print(f'F1-score: {cfm_threshold.f1_score_}')
    print(f'Accuracy: {cfm_threshold.accuracy_}')
    print(f'Precision: {cfm_threshold.precision_}')
    print(f'Recall: {cfm_threshold.recall_}')
    print(f'False Omission Rate: {cfm_threshold.false_omission_rate_}')
    print(f'False Discovery Rate: {cfm_threshold.false_discovery_rate_}')
    print(f'False Negative Rate: {cfm_threshold.false_negative_rate_}')
    print(f'False Positive Rate: {cfm_threshold.false_positive_rate_}')
    print(f'Negative Predictive Value: {cfm_threshold.negative_predictive_value_}')
    print(f'Specificity: {cfm_threshold.specificity_}')
    print(f'True Positives: {cfm_threshold.tp_}')
    print(f'False Positives: {cfm_threshold.fp_}')
    print(f'True Negatives: {cfm_threshold.tn_}')
    print(f'False Negatives: {cfm_threshold.fn_}')
print(f'Plot precision recall curve: {xgbc_metric.plot_precision_recall_curve_}')
print(f'Plot ROC: {xgbc_metric.plot_roc_curve_}')

# from myml.metrics.classification import ConfusionMatrix
# import numpy as np

# a = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
# b = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1])
# a = np.array([1, 0, 2, 0, 1, 2, 1, 0, 2, 0, 1, 2, 1])
# b = np.array([1, 1, 2, 0, 0, 1, 1, 0, 2, 1, 1, 0, 0])
# cfm = ConfusionMatrix(a, b)
# print(f'Confusion Matrix: {cfm.table_}')
# print(f'False positive: {cfm.fp_}')
# print(f'False negative: {cfm.fn_}')
# print(f'True positive: {cfm.tp_}')
# print(f'True negative: {cfm.tn_}')
# print(f'Recall: {cfm.recall_}')
# print(f'Specificity: {cfm.specificity_}')
# print(f'Precision: {cfm.precision_}')
# print(f'Negative predictive value: {cfm.negative_predictive_value_}')
# print(f'False positive rate: {cfm.false_positive_rate_}')
# print(f'False Negative rate: {cfm.false_negative_rate_}')
# print(f'False Discovery rate: {cfm.false_discovery_rate_}')
# print(f'False Omission rate: {cfm.false_omission_rate_}')
# print(f'Accuracy: {cfm.accuracy_}')

