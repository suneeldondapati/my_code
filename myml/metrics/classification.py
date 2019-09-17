import pandas as pd
import numpy as np
import sklearn.metrics as mt
from abc import ABCMeta, abstractmethod


__all__ = [
    'BinaryClassificationMetrics',
    'ConfusionMatrix',
]


class ClassificationMetrics(metaclass=ABCMeta):
    """Classification Metrics

    Arguments:
    ----------
    clf: Classifier
        Trained classification model
    X: pandas.DataFrame
        IDV data used for modelling
    y: pandas.Series
        DV
    """

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.predicted_proba = model.predict_proba(self.X)

    @abstractmethod
    def confusion_matrix(self, **kwargs):
        pass


class BinaryClassificationMetrics(ClassificationMetrics):
    """ Binary Classification Metrics

    Parameters:
    -----------
    X: pandas.DataFrame
        Test IDVs
    y: pandas.Series
        Test DV
    labels: dict, optional (default = {0: 0, 1: 1})
        labels for y, eg: {0: 'negative', 1: 'positive'}
    """

    __thresholds: np.array
    X: pd.DataFrame
    y: pd.Series

    def __init__(self, clf, X, y, labels: dict = None):
        super().__init__(clf, X, y)
        self.labels = labels
        self.predicted_proba = self.predicted_proba[:, 1]
        self.__fpr, self.__tpr, self.__thresholds = mt.roc_curve(self.y, self.predicted_proba)
        self._threshold = 0.5
        self._y_hat = None
        self.cfm = None
        self._gains_table = pd.DataFrame()

    @property
    def auc_(self):
        """Area Under the Curve"""
        return mt.auc(self.__fpr, self.__tpr)

    @property
    def plot_roc_curve_(self):
        raise NotImplementedError

    @property
    def plot_precision_recall_curve_(self):
        raise NotImplementedError

    @property
    def gains_table_(self):
        if self._gains_table.empty:
            df = pd.DataFrame()
            df['y'] = self.y
            df['pred_prob'] = self.predicted_proba
            try:
                df['Decile'] = pd.qcut(df['pred_prob'], 10, labels=False)
            except ValueError:
                df['Decile'] = pd.qcut(df['pred_prob'].rank(method='first'), 10, labels=False)
            df = df.rename_axis('unique_id').reset_index()
            lift_df = (df
                       .groupby(['Decile', 'y'])['unique_id']
                       .count()
                       .unstack('y')
                       .sort_index(ascending=False))
            lift_df.index = np.arange(10)
            gains_df = pd.DataFrame()
            kwargs = {'Decile': lift_df.index + 1,
                      'No. of Observations': lift_df.sum(axis=1),
                      'Number of Targets': lift_df[1],
                      'Cumulative Targets': lift_df[1].cumsum(),
                      '% of Targets': lift_df[1] / lift_df[1].sum() * 100,
                      'Gain': lift_df[1].cumsum() / lift_df[1].sum() * 100,
                      'Random Targets': lift_df[1].sum() / 10}
            gains_df = gains_df.assign(**kwargs)
            gains_df['Lift'] = lift_df[1] / gains_df['Random Targets']
            gains_df['Cumulative Lift'] = gains_df['Lift'].cumsum()
            # gains_df['Cumulative Lift'] = (gains_df['Cumulative Targets'] /
            #                                (lift_df[1].sum() * (lift_df.index + 1) / 10))
            gains_sum = pd.DataFrame({}, columns=gains_df.columns, index=[0])
            kwargs = {'Decile': 'Total',
                      'No. of Observations': lambda x: x['No. of Observations'].sum(),
                      'Number of Targets': lambda x: x['Number of Targets'].sum()}
            gains_sum = gains_sum.assign(**kwargs)
            gains_table = pd.concat([gains_df, gains_sum], ignore_index=True)
            gains_table.fillna('', inplace=True)
            self._gains_table = gains_table
        return self._gains_table

    @property
    def lift_score(self):
        return self.gains_table_['Lift'].iloc[0]

    @property
    def y_hat(self):
        """Predicted values"""
        self._y_hat = (self.predicted_proba > self.threshold).astype(bool)
        return self._y_hat

    @property
    def threshold(self):
        """Threshold chosen"""
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        """Threshold to set"""
        if not (0 < threshold < 1):
            raise ValueError("Invalid 'threshold'. It should be in range of (0, 1)")
        self._threshold = threshold

    def confusion_matrix(self, threshold=None):
        """Confusion matrix for the given threshold

        Parameters:
        -----------
        threshold: float
            Range (0, 1)

        Returns:
            Confusion matrix object
        """
        self.threshold = self.threshold if not threshold else threshold
        confusion_matrix = ConfusionMatrix(self.y, self.y_hat)
        if not self.labels:
            self.labels = {}
        label_0 = self.labels.get(0, 0)
        label_1 = self.labels.get(1, 1)
        table = pd.DataFrame(confusion_matrix.table_,
                             columns=[label_0, label_1],
                             index=[label_0, label_1])
        self.cfm = table.rename_axis(f'threshold ({self.threshold})').reset_index()
        return confusion_matrix


class ConfusionMatrix:

    def __init__(self, y, y_hat):
        self.y = y
        self.y_hat = y_hat
        self.__binary_y = self.__is_y_binary()
        self._cfm = np.array([])
        self._fp = np.array([])
        self._fn = np.array([])
        self._tp = np.array([])
        self._tn = np.array([])

    def __is_y_binary(self):
        return len(np.unique(self.y_hat)) == 2

    @property
    def table_(self):
        """Confusion Matrix"""
        if not self._cfm.size:
            self._cfm = mt.confusion_matrix(self.y, self.y_hat)
        return self._cfm

    def __cfm_metric(self, value):
        if self.__binary_y:
            return value[1]
        return value

    @property
    def fp_(self):
        """False Positives"""
        if not self._fp.size:
            fp = self.table_.sum(axis=0) - np.diag(self.table_)
            self._fp = self.__cfm_metric(fp)
        return self._fp

    @property
    def fn_(self):
        """False Negatives"""
        if not self._fn.size:
            fn = self.table_.sum(axis=1) - np.diag(self.table_)
            self._fn = self.__cfm_metric(fn)
        return self._fn

    @property
    def tp_(self):
        """True Positives"""
        if not self._tp.size:
            tp = np.diag(self.table_)
            self._tp = self.__cfm_metric(tp)
        return self._tp

    @property
    def tn_(self):
        """True Negatives"""
        if not self._tn.size:
            self._tn = self.table_.sum() - (self.fp_ + self.fn_ + self.tp_)
        return self._tn

    @property
    def recall_(self):
        """Sensitivity, Recall, Hit rate, or True Positive Rate"""
        return self.tp_ / (self.tp_ + self.fn_)

    @property
    def specificity_(self):
        """Specificity, Selectivity, or True Negative Rate"""
        return self.tn_ / (self.tn_ + self.fp_)

    @property
    def precision_(self):
        """Precision or Positive Predictive Value"""
        return self.tp_ / (self.tp_ + self.fp_)

    @property
    def negative_predictive_value_(self):
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
    def false_omission_rate_(self):
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

    def __getattribute__(self, item):
        if item in {'recall_', 'specificity_', 'precision_', 'negative_predictive_value_',
                    'false_negative_rate_', 'false_positive_rate_', 'false_discovery_rate_',
                    'false_omission_rate_', 'accuracy_', 'f1_score_'}:
            value = super(ConfusionMatrix, self).__getattribute__(item)
            return np.round(value * 100, 2)
        return super(ConfusionMatrix, self).__getattribute__(item)


if __name__ == '__main__':
    pass








