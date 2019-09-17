import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
import xgboost as xgb

from ..metrics import BinaryClassificationMetrics


__all__ = [
    'XGBClassifier',
]


SEED = 1310


class Classifier(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, params):
        self.params = params
        self.feature_names = None
        self.y_col_name = None

    @abstractmethod
    def fit(self, X, y):
        """Train XGBoost Classifier

        Parameters:
        -----------
        X: pandas.DataFrame
            IDV data
        y: pandas.Series
            DV data

        Returns:
        --------
        Trained model
        """
        X = X.copy()
        y = y.copy()
        self.feature_names = X.columns.tolist()
        self.y_col_name = y.name

    @abstractmethod
    def classification_metric(self, X, y):
        X = X.copy()
        y = y.copy()


class XGBClassifier(Classifier):
    """XGBClassifier

    Parameters:
    -----------
    params: dict
        Model parameters
    """
    def __init__(self, params=None):
        super().__init__(self.__set_xgb_params(params))
        self.model = None
        self._feature_importance_ = None

    @staticmethod
    def __set_xgb_params(xgb_params_in):
        xgb_params_default = xgb.XGBClassifier().get_xgb_params()
        if not xgb_params_in:
            return xgb_params_default
        if not isinstance(xgb_params_in, dict):
            raise TypeError("Parameters passed to classifier should be of type 'dict'.")
        unknown_xgb_params = set(xgb_params_in.keys()) - set(xgb_params_default.keys())
        if unknown_xgb_params:
            raise AttributeError(f'Unknown parameters for XGBoost model: {str(unknown_xgb_params)[1:-1]}.')
        xgb_params_out = {}
        for param, value in xgb_params_default.items():
            xgb_params_out[param] = xgb_params_in.get(param, xgb_params_default[param])
        return xgb_params_out

    def fit(self, X, y):
        super().fit(X, y)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)

    @property
    def feature_importance_(self):
        """XGBoost Classifier feature importance
        """
        if self._feature_importance_:
            return self._feature_importance_
        if not self.model:
            raise AttributeError('Fit the model before calling feature_importance')
        # imp_vals = self.model.booster().get_fscore()
        imp_vals = self.model.get_booster().get_fscore()
        imp_dict = {self.feature_names[i]: float(imp_vals.get(feature, 0.))
                    for i, feature in enumerate(self.feature_names)}
        total = sum(imp_dict.values())
        feature_imp = {feature: (imp/total) for feature, imp in imp_dict.items()}
        feature_imp_sorted = sorted(feature_imp.items(), key=lambda kv: kv[1], reverse=True)
        self._feature_importance_ = pd.DataFrame(feature_imp_sorted, columns=['IDV', 'Importance'])
        return self._feature_importance_

    def classification_metric(self, X, y, labels=None):
        super().classification_metric(X, y)
        if not self.model:
            raise AttributeError('Fit the model before calling classification_metric')
        bclf_metrics = BinaryClassificationMetrics(self.model, X.copy(), y.copy(), labels=labels)
        return bclf_metrics

































