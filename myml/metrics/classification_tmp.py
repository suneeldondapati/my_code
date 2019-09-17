#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Filename: classification_tmp.py
Date: 2019-08-30 20:52
Project: my_code
AUTHOR: Suneel Dondapati
"""
import pandas as pd
import numpy as np
import sklearn.metrics as mt
from abc import ABCMeta, abstractmethod


class Classification:
    pass

class BinaryClassification(Classification):

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __get_predicted_classes(self, pred_probs):
        if not (0 < self.threshold < 1):
            raise ValueError("Invalid 'threshold'. It should be in range of (0, 1)")
        return (pred_probs[:, 1] > self.threshold).astype(bool)

    def __cfm_metric(self, fp):
        return fp[1]


class MulticlassClassification(Classification):

    def __init__(self):
        pass

    def __get_predicted_classes(self, pred_probs):
        return np.argmax(pred_probs, axis=1)

    def __cfm_metric(self, fp):
        return fp


class ConfusionMatrix:

    _classification_type: Classification

    def __init__(self, y, pred_probs, classification_type=None):
        self.y = y
        self.pred_probs = pred_probs
        if not classification_type:
            ValueError("ClassificationType class is not supplied to ConfusionMatrix")
        self._classification_type = classification_type
        self.y_hat = self.__get_predicted_classes()
        self._cfm = None
        self._fp = None
        self._fn = None
        self._tp = None
        self._tn = None

    def __get_predicted_classes(self):
        return self._classification_type.__get_predicted_classes(self.pred_probs)

    @property
    def table_(self):
        """Confusion Matrix"""
        if not self._cfm:
            self._cfm = mt.confusion_matrix(self.y, self.y_hat)
        return self._cfm

    @property
    def fp_(self):
        """False Positives"""
        if not self._fp:
            fp = self.table_.sum(axis=0) - np.diag(self.table_)
            self._fp = self._classification_type.__cfm_metric(fp)
        return self._fp

    @property
    def fn_(self):
        """False Negatives"""
        if not self._fn:
            fn = self.table_.sum(axis=1) - np.diag(self.table_)
            self._fn = self._classification_type.__cfm_metric(fn)
        return self._fn

    @property
    def tp_(self):
        """True Positives"""
        if not self._tp:
            tp = np.diag(self.table_)
            self._tp = self._classification_type.__cfm_metric(tp)
        return self._tp

    @property
    def tn_(self):
        """True Negatives"""
        if not self._tn:
            tn = self.table_.sum() - (self.fp_+ self.fn_ + self.tp_)
            self._tn = self._classification_type.__cfm_metric(tn)
        return self._tn

    @property
    def recall_(self):
        """Sensitivity, Recall, Hit rate, or True Positive Rate"""
        return self.tp_ / (self.tp_ + self.tn_)

    @property
    def specificity_(self):
        """Specificity, Selectivity, or True Negative Rate"""
        return self.tn_ / (self.tn_ + self.fp_)

    @property
    def precision_(self):
        """Precision or Positive Predictive Value"""
        return self.tp_ / (self.tp_ + self.fp_)

    @property
    def negative_predictive_value(self):
        """Negative Predictive Value"""
        return self.tn_ / (self.tn_ + self.fn_)

    @property
    def false_negative_rate_(self):
        """False Negative Rate or Miss Rate"""
        return self.fn_ / (self.fn_ + self.tp_)

    @property
    def false_positive_rate_(self):
        """False Positive Rate or Fall-out"""
        return self.fp_ / (self.fp_ + self.tn_)

    @property
    def false_discovery_rate_(self):
        """False Discovery Rate"""
        return self.fp_ / (self.fp_ + self.tp_)

    @property
    def false_omission_rate(self):
        """False Omission Rate"""
        return self.fn_ / (self.fn_ + self.tn_)

    @property
    def accuracy_(self):
        """Accuracy"""
        return (self.tp_ + self.tn_) / (self.tp_ + self.tn_ + self.fp_ + self.fn_)

    @property
    def f1_score_(self):
        """F1 Score"""
        return (2 * self.tp_) / (2 * self.tp_ + self.fp_ + self.fn_)




class BinaryClassificationMetrics(ClassificationMetrics):

    __thresholds: np.array

    def __init__(self, clf, X, y, pos_label: str = None):
        super().__init__(clf, X, y)
        self.pos_label = pos_label
        self.y = self.__to_binary(self.y)
        self.prediction_probs = self.prediction_probs[:, 1]
        self.__fpr, self.__tpr, self.__thresholds = mt.roc_curve(self.y, self.prediction_probs)
        self._threshold = None

    def __to_binary(self, y):
        if not self.__binary_y and not self.pos_label:
            raise AttributeError("Either pass binary values for 'y' or, "
                                 "label for positive class using 'pos_label' attribute")
        if not self.__binary_y and self.pos_label:
            y = pd.Series(np.where(self.y == self.pos_label, 1, 0),
                          name=self.y.name)
        return y

    @property
    def __binary_y(self):
        return sorted(self.y) == [0, 1]

    @property
    def auc(self):
        return mt.auc(self.__fpr, self.__tpr)

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if not value:
            self._threshold = 0.5
        else:
            if not (0 < value < 1):
                raise ValueError("Threshold should be between '0' and '1'")
            self._threshold = value

    @property
    def y_hat(self):
        return (self.prediction_probs > self.threshold).astype(bool)

    def confusion_matrix(self, threshold=None, use_optimal_threshold=False):
        super(BinaryClassificationMetrics, self).confusion_matrix()
        if use_optimal_threshold and threshold:
            raise ValueError("When value for 'threshold' is passed, "
                             "optimal_threshold should be 'False'")
        if use_optimal_threshold:
            optimal_idx = np.argmax(self.__tpr - self.__fpr)
            threshold = self.__thresholds[optimal_idx]
        self.threshold = threshold
        cfm = mt.confusion_matrix(self.y, self.y_hat)
        cfm = pd.DataFrame(cfm, columns=[0, 1],
                           index=[0, 1])
        cfm = cfm.rename_axis(f'threshold ({self.threshold})')
        if self.pos_label:
            pos_label = [self.pos_label]
            neg_label = [label for label in self.classification_labels
                         if label not in pos_label][0]
            cfm.rename({0: neg_label, 1: self.pos_label}, axis=1, inplace=True)
            cfm.rename({0: neg_label, 1: self.pos_label}, axis=0, inplace=True)
            return cfm
        return cfm

