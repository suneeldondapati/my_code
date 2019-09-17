# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:39:20 2019

@author: suneel.dondapati
"""


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


df_performance_metrics = pd.DataFrame(columns=[
                                                # **** Model Name ****
                                                'Model_Name', 
    
                                                # **** Reason for transformation ****
                                                'col_action',
            
                                                # **** Training Scores ****
                                                'Train_f1_score', 
                                                'Train_precision', 
                                                'Train_recall_sensitivity_tpr', 
                                                'Train_specificity_tnr', 
            
                                                # **** Validation Scores ****
                                                'Valid_f1_score', 
                                                'Valid_precision', 
                                                'Valid_recall_sensitivity_tpr', 
                                                'Valid_specificity_tnr', 
            
                                                # **** Test Scores ****
                                                'Test_f1_score', 
                                                'Test_precision', 
                                                'Test_recall_sensitivity_tpr', 
                                                'Test_specificity_tnr',
                                        ]
                                      )

def split_list_scores(results_Df: pd.DataFrame):
    cols = results_Df.columns[2:].copy()

    for i in cols:
        results_Df[[i+'_0',i+'_1',i+'_2']] = pd.DataFrame(results_Df.loc[:,i].values.tolist(), index= results_Df.index)
        
    return results_Df.reset_index(drop=True)


class Confusion_Matrix_Scores:
    def __init__(
                    self, 
                    Y_True: list, 
                    Y_Predict: list
    ):
        # Set Input Variables [Data and Model]
        self.Y_True = Y_True
        self.Y_Predict = Y_Predict
        
        # Initialize Confusion Matrix & Score Values
        max_shape_of_conf_matrix = max(len(set(self.Y_True)), len(set(self.Y_Predict )))
        self.cnf_matrix = np.zeros(shape=(max_shape_of_conf_matrix, max_shape_of_conf_matrix))
        
        self.FP = 0 ## False Positive
        self.FN = 0 ## False Negative
        self.TP = 0 ## True Positive
        self.TN = 0 ## True Negative
        
        self.TPR = 0 ## Sensitivity, hit rate, recall, or true positive rate
        self.TNR = 0 ## Specificity or true negative rate
        self.PPV = 0 ## Precision or positive predictive value
        self.NPV = 0 ## Negative predictive value
        self.FPR = 0 ## Fall out or false positive rate
        self.FNR = 0 ## False negative rate
        self.FDR = 0 ## False discovery rate
        self.ACC = 0 ## Overall accuracy
        self.F1  = 0 ## F1 Score
        
        # Generate All the Scores
        self.generate_scores()
        
    # Generate Confusion Matrix, Precision, Recall
    def generate_scores(self):
        self.cnf_matrix = confusion_matrix(self.Y_True, self.Y_Predict)
        self.FP = self.cnf_matrix.sum(axis=0) - np.diag(self.cnf_matrix)  
        self.FN = self.cnf_matrix.sum(axis=1) - np.diag(self.cnf_matrix)
        self.TP = np.diag(self.cnf_matrix)
        self.TN = self.cnf_matrix.sum() - (self.FP + self.FN + self.TP)

        self.FP = self.FP.astype(float)
        self.FN = self.FN.astype(float)
        self.TP = self.TP.astype(float)
        self.TN = self.TN.astype(float)

        # Sensitivity, hit rate, recall, or true positive rate
        self.TPR = self.TP/(self.TP+self.FN)
        self.TPR[np.isnan(self.TPR)]=0
        
        # Specificity or true negative rate
        self.TNR = self.TN/(self.TN+self.FP)
        self.TNR[np.isnan(self.TNR)]=0
        
        # Precision or positive predictive value
        self.PPV = self.TP/(self.TP+self.FP)
        self.PPV[np.isnan(self.PPV)]=0
        
        # Negative predictive value
        self.NPV = self.TN/(self.TN+self.FN)
        self.NPV[np.isnan(self.NPV)]=0
        
        # Fall out or false positive rate
        self.FPR = self.FP/(self.FP+self.TN)
        self.FPR[np.isnan(self.FPR)]=0
        
        # False negative rate
        self.FNR = self.FN/(self.TP+self.FN)
        self.FNR[np.isnan(self.FNR)]=0
        
        # False discovery rate
        self.FDR = self.FP/(self.TP+self.FP)
        self.FDR[np.isnan(self.FDR)]=0
        
        # Overall accuracy
        self.ACC = (self.TP+self.TN)/(self.TP+self.FP+self.FN+self.TN)
        self.ACC[np.isnan(self.ACC)]=0
        
        # F1 Score
        self.F1 = 2*((self.PPV*self.TPR)/(self.PPV+self.TPR))
        self.F1[np.isnan(self.F1)]=0
        
        
    # Get Sensitivity, hit rate, recall, or true positive rate
    def get_Sensitivity_Recall_TPR(self):
        return self.TPR
        
        
    # Get Specificity or true negative rate
    def get_Specificity_TNR(self):
        return self.TNR
        
        
    # Get Precision or positive predictive value
    def get_Precision_PPR(self):
        return self.PPV  
        
        
    # Get Negative predictive value
    def get_NPR(self):
        return self.NPR    
        
        
    # Get Fall out or false positive rate
    def get_Fallout_FPR(self):
        return self.FPR
        
    # Get False negative rate
    def get_FNR(self):
        return self.FNR
        
        
    # Get False discovery rate
    def get_FDR(self):
        return self.FDR
        
        
    # Get Overall accuracy
    def get_Accuracy_ACC(self):
        return self.ACC
    
    
    # Get F1 Score
    def get_F1_Score_F1(self):
        return self.F1
    
    
class Fit_Transform_Models:
    def __init__(
                    self, 
                    X_Train: pd.DataFrame, 
                    Y_Train: pd.DataFrame, 
                    X_Valid: pd.DataFrame,
                    Y_Valid: pd.DataFrame,
                    X_Test: pd.DataFrame,
                    Y_Test: pd.DataFrame,
                    model: BaseEstimator,
                    modelName: str,
                    col_action: str
    ):
        # Set Input Variables [Data and Model]
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.X_Valid = X_Valid
        self.Y_Valid = Y_Valid
        self.X_Test = X_Test
        self.Y_Test = Y_Test
        self.model = model
        self.modelName = modelName
        self.col_action = col_action
        
        # Initialize a new Performance_Metrics dataframe
        self.results_Df = df_performance_metrics.copy(deep=True)
        
        # Initialize Y_Predict & Y_Predict_Prob
        self.Y_Train_Predict = Y_Train.copy()
        self.Y_Valid_Predict = Y_Valid.copy()
        self.Y_Test_Predict = Y_Train.copy()
        
    # Fits the input Model
    def fit(self, X, Y):
        self.model = self.model.fit(X, Y)
        
    # Predict Class for X and Returns  the same
    def predict(self, X):
        return self.model.predict(X)
        
    # Predict Probability Score for X and Returns the same
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    # Fits X and Returns Y
    def get_scores(self):
        self.fit(self.X_Train, self.Y_Train)
        self.Y_Train_Predict = self.predict(self.X_Train)
        
        self.fit(self.X_Valid, self.Y_Valid)
        self.Y_Valid_Predict = self.predict(self.X_Valid)
        
        self.fit(self.X_Test, self.Y_Test)
        self.Y_Test_Predict = self.predict(self.X_Test)
        
        confusion_Matrix_Scores_Train = Confusion_Matrix_Scores(self.Y_Train, self.Y_Train_Predict)
        confusion_Matrix_Scores_Valid = Confusion_Matrix_Scores(self.Y_Valid, self.Y_Valid_Predict)
        confusion_Matrix_Scores_Test = Confusion_Matrix_Scores(self.Y_Test, self.Y_Test_Predict)
        
        self.results_Df = self.results_Df.append(
                            {
                                # **** Model Name ****
                                'Model_Name': self.modelName,
                                
                                # **** Reason for Transformation ****
                                'col_action' : self.col_action,
                                
                                # **** Training Scores ****
                                'Train_f1_score': confusion_Matrix_Scores_Train.get_F1_Score_F1(), 
                                'Train_precision': confusion_Matrix_Scores_Train.get_Precision_PPR(), 
                                'Train_recall_sensitivity_tpr': confusion_Matrix_Scores_Train.get_Sensitivity_Recall_TPR(), 
                                'Train_specificity_tnr': confusion_Matrix_Scores_Train.get_Specificity_TNR(), 
                                
                                # **** Validation Scores ****
                                'Valid_f1_score': confusion_Matrix_Scores_Valid.get_F1_Score_F1(),  
                                'Valid_precision': confusion_Matrix_Scores_Valid.get_Precision_PPR(),  
                                'Valid_recall_sensitivity_tpr': confusion_Matrix_Scores_Valid.get_Sensitivity_Recall_TPR(),  
                                'Valid_specificity_tnr': confusion_Matrix_Scores_Valid.get_Specificity_TNR(), 

                                # **** Test Scores ****
                                'Test_f1_score': confusion_Matrix_Scores_Test.get_F1_Score_F1(),  
                                'Test_precision': confusion_Matrix_Scores_Test.get_Precision_PPR(),  
                                'Test_recall_sensitivity_tpr': confusion_Matrix_Scores_Test.get_Sensitivity_Recall_TPR(),  
                                'Test_specificity_tnr': confusion_Matrix_Scores_Test.get_Specificity_TNR()
                            }, 
                            ignore_index=True
        )
        
        return self.results_Df
    
    def run_models(
                col_action: str,
                loc_X_Train: pd.DataFrame,
                loc_Y_Train: pd.DataFrame,
                loc_X_Valid: pd.DataFrame,
                loc_Y_Valid: pd.DataFrame,
                loc_X_Test: pd.DataFrame,
                loc_Y_Test: pd.DataFrame
              ):
    logistic_regression = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    decision_tree = DecisionTreeClassifier(random_state=0)
    naive_bayes = GaussianNB()
    random_forest = RandomForestClassifier(criterion = 'entropy')
    xg_boost = XGBClassifier()
    
    models_dict = {
                "logistic_regression": logistic_regression,
                "decision_tree": decision_tree,
                "naive_bayes": naive_bayes,
                "random_forest": random_forest,
                "xg_boost": xg_boost
            }
    
    local_results_Df = df_performance_metrics.copy(deep=True)
    for k, v in models_dict.items():
        local_results_Df = local_results_Df.append(Fit_Transform_Models(
                                    loc_X_Train, loc_Y_Train, 
                                    loc_X_Valid, loc_Y_Valid,
                                    loc_X_Test, loc_Y_Test,
                                    v,
                                    k,
                                    col_action
                ).get_scores())
        
    local_results_Df = split_list_scores(local_results_Df)
    
    Test_recall_sensitivity_tpr = [col for col in local_results_Df if ('Test_recall_sensitivity_tpr_' in col)]
    Valid_recall_sensitivity_tpr = [col for col in local_results_Df if ('Valid_recall_sensitivity_tpr_' in col)]
    Train_recall_sensitivity_tpr = [col for col in local_results_Df if ('Train_recall_sensitivity_tpr_' in col)]

    Test_specificity_tnr = [col for col in local_results_Df if ('Test_specificity_tnr_' in col)]
    Valid_specificity_tnr = [col for col in local_results_Df if ('Valid_specificity_tnr_' in col)]
    Train_specificity_tnr = [col for col in local_results_Df if ('Train_specificity_tnr_' in col)]

    Test_f1_score = [col for col in local_results_Df if ('Test_f1_score_' in col)]
    Valid_f1_score = [col for col in local_results_Df if ('Valid_f1_score_' in col)]
    Train_f1_score = [col for col in local_results_Df if ('Train_f1_score_' in col)]

    Test_precision = [col for col in local_results_Df if ('Test_precision_' in col)]
    Valid_precision = [col for col in local_results_Df if ('Valid_precision_' in col)]
    Train_precision = [col for col in local_results_Df if ('Train_precision_' in col)]

    local_results_Df = local_results_Df.sort_values(by=(Test_f1_score + Valid_f1_score + Train_f1_score + Test_recall_sensitivity_tpr + Valid_recall_sensitivity_tpr + Train_recall_sensitivity_tpr + Test_precision + Valid_precision + Train_precision + Test_specificity_tnr + Valid_specificity_tnr + Train_specificity_tnr + ["Model_Name"]), ascending=False)
        
    return local_results_Df