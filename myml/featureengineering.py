# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:47:30 2019

@author: suneel.dondapati
"""


import numpy as np
import pandas as pd
import re
import traceback
from scipy.stats import boxcox, kurtosis, skew
from typing import Tuple


class Transform:
    """Transform the columns of the input dataframe
    
    Example:
    --------
    >>> transformation = {
    ...     'Assets_through_Direct_Channels': Transform.boxcox_of,
    ...     'Assets_through_basic_accounts': Transform.tanh_of,
    }
    >>> transform_obj = Transform(df)
    >>> for column, transform in transformation.items():
    ...     transform(transform_obj, column)
    ...     print(f'Transformed {column}')
    Transformed Assets_through_Direct_Channels
    Transformed Assets_through_basic_accounts
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def log_of_one_plus(self, column):
        """
        log(1 + column)
        """
        self.df[f'log_of_one_plus_{column}'] = self.df[column].apply(lambda x: np.log(1+x))
        
    def square_root_of(self, column):
        """
        sqrt(column)
        """
        self.df[f'square_root_of_{column}'] = self.df[column].apply(lambda x: np.sqrt(x))
        
    def cube_root_of(self, column):
        """
        column^(1/3)
        """
        self.df[f'cube_root_of_{column}'] = self.df[column].apply(lambda x: x**(1./3.))
    
    def sigmoid_of(self, column):
        """
        sigmoid(column) = 1 / (1 + exp(-column))
        """
        self.df[f'sigmoid_of_{column}'] = self.df[column].apply(lambda x: 1./(1+np.exp(-x)))
    
    def tanh_of(self, column):
        """
        tanh(column) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        """
        self.df[f'tanh_of_{column}'] = self.df[column].apply(lambda x: np.tanh(x))
        
    def reciprocal_of_one_plus(self, column):
        """
        1/1+column
        """
        self.df[f'reciprocal_of_one_plus_{column}'] = self.df[column].apply(lambda x: 1./(1+x))
    
    def exponent_of(self, column):
        """
        exp(column)
        """
        self.df[f'exponent_of_{column}'] = self.df[column].apply(lambda x: np.exp(x))
    
    def boxcox_of(self, column):
        """
        y = (column**lambda - 1) / lambda, for lambda > 0
            log(column)                  , for lambda = 0
        """
        if self.df[column].min() == 0:
            self.df[column] = self.df[column].apply(lambda x: x+1)
        self.df[f'boxcox_of_{column}'], self.lambda_choosen = boxcox(self.df[column].values)
		
		
def transform_and_calculate_skewness_kurtosis(df: pd.DataFrame, 
                                              columns: list) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Get the skewness and kurtosis of the columns mentioned.
    If none of the columns are mentioned calculate for all the
    columns.
    
    Arguments:
    ----------
    df: pd.DataFrame
        Input dataframe
    columns:
        Columns on which transformations to be applied and
        skewness, kurtosis values are to be calculated
    
    Returns:
    --------
    transformed_df: pd.DataFrame
        Transformed dataframe
    skewness_kurtosis_: pd.DataFrame
        DataFrame with sorted skewness and kurtosis values for 
		each input variable
    lambda_choosen_for_: dict
        Lambda values choosen for box-cox transformation
		
    Example:
    --------
    >>> transformed_df, sk_df, lambdas_choosen = transform_and_calculate_skewness_kurtosis(df, ['Tenure_with_bank', 
    ...																							'Total_Asset_Value'])
    """
    lambda_choosen_for_, skewness_vals, kurtosis_vals = {}, {}, {}
    if not columns:
        columns = df.columns
    passed_columns_set = set(columns)
    df_columns_set = set(df.columns)
    if not passed_columns_set.issubset(df_columns_set):
        unknown_columns = df_columns_set - passed_columns_set
        raise ValueError('{list(unknown_columns)} columns which are mentioned are not in dataframe')
    for column in columns:
        print(column)
        if df[column].dtype.name != 'category':
            transform = Transform(df)
            transform.cube_root_of(column)
            transform.exponent_of(column)
            transform.log_of_one_plus(column)
            transform.reciprocal_of_one_plus(column)
            transform.sigmoid_of(column)
            transform.square_root_of(column)
            transform.tanh_of(column)
            transform.boxcox_of(column)
            lambda_choosen_for_.update({column: transform.lambda_choosen})
            kurtosis_vals.update({(column, column): kurtosis(df[column])})
            kurtosis_vals.update({(column, f'{transformation}_{column}'): kurtosis(df[f'{transformation}_{column}']) 
                             for transformation in ['cube_root_of', 'exponent_of', 'log_of_one_plus', 
                                            'reciprocal_of_one_plus', 'sigmoid_of', 'square_root_of',
                                            'square_root_of', 'tanh_of', 'boxcox_of']})
            skewness_vals.update({(column, column): skew(df[column])})
            skewness_vals.update({(column, f'{transformation}_{column}'): skew(df[f'{transformation}_{column}']) 
                             for transformation in ['cube_root_of', 'exponent_of', 'log_of_one_plus', 
                                            'reciprocal_of_one_plus', 'sigmoid_of', 'square_root_of',
                                            'square_root_of', 'tanh_of', 'boxcox_of']})
    transformed_df = df.copy()
    skewness_kurtosis_df = pd.concat([pd.Series(skewness_vals, name='Skewness'), 
                                      pd.Series(kurtosis_vals, name='Kurtosis')], axis=1)
    skewness_kurtosis_df['abs_Skewness'] = skewness_kurtosis_df['Skewness'].map(abs)
    skewness_kurtosis_df['abs_Kurtosis'] = skewness_kurtosis_df['Kurtosis'].map(abs)
    skewness_kurtosis_ = pd.DataFrame()
    for grp_name, grp in skewness_kurtosis_df.groupby(level=0):
        grp = grp.sort_values(by=['abs_Skewness', 'abs_Kurtosis'])
        skewness_kurtosis_ = pd.concat([skewness_kurtosis_, grp])
    return (transformed_df, skewness_kurtosis_, lambda_choosen_for_)
	
	
def continuous_value_binning(dependent_variable: pd.Series,
                             independent_variable: pd.Series,
                             max_bins=20,
                             min_bins=2) -> pd.DataFrame:
    """Identifies optimal number of bins and calculate 
    Weight of Evidence Encoding (WOE) for each bin and 
    Information Value (IV) for the given independent variable.
    
    Arguments:
    ----------
    dependent_variable: pd.Series
        Target variable
    independent_variable: pd.Series
        Independent variable
    max_bins: int
        Maximum bins to be created for the given independent variable.
        Number of bins will be decreased to acceptable value based on
        Spearman's rank correlation value.
    min_bins: int
        Minimum bins to be created, if Spearman's rank correlation value
        is not equal to 1 for bins in the range of [2, max_bins] 
    
    Returns:
    --------
    binned_rows_df: pd.DataFrame
        Calculated WOE for each bin and 
        IV for the given independent variable
    """
    import pandas.core.algorithms as algos
    from scipy.stats import spearmanr
    
    
    df = pd.DataFrame({"X": independent_variable, 
                       "Y": dependent_variable})
    null_rows = df[df.X.isnull()]
    non_null_rows = df[df.X.notnull()]
    # Spearman's rank correlation coefficient
    r = 0
    while np.abs(r) < 1:
        try:
            tmp_df = pd.DataFrame({"X": non_null_rows.X, 
                                   "Y": non_null_rows.Y, 
                                   "Bucket": pd.qcut(non_null_rows.X, max_bins)})
            grouped_tmp_df = tmp_df.groupby('Bucket', as_index=True)
            # r = Spearman's rank correlation coefficient
            # p = Two sided p-value for a hypothesis test whose null hypothesis
            #     is that, corr(X, Y) = 0 (un-correlated)
            r, p = spearmanr(grouped_tmp_df.mean().X, grouped_tmp_df.mean().Y)
            max_bins -= 1
            # print(f"Calculated spearman's r: {r}")
        except Exception as e:
            max_bins -= 1

    if len(grouped_tmp_df) == 1:         
        bins = algos.quantile(non_null_rows.X, 
                              np.linspace(0, 1, min_bins+1))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        tmp_df = pd.DataFrame({"X": non_null_rows.X,
                               "Y": non_null_rows.Y,
                               "Bucket": pd.cut(non_null_rows.X,
                                                np.unique(bins),
                                                include_lowest=True)})
        grouped_tmp_df = tmp_df.groupby('Bucket', as_index=True)
    
    binned_non_null_rows_df = pd.DataFrame({},index=[])
    # Minimum value for each bin
    binned_non_null_rows_df["min_value"] = grouped_tmp_df.min().X
    # Maximum value for each bin
    binned_non_null_rows_df["max_value"] = grouped_tmp_df.max().X
    # Count of values for each bin
    binned_non_null_rows_df["count"] = grouped_tmp_df.count().Y
    # Number of churned customers in each bin
    binned_non_null_rows_df["churn"] = grouped_tmp_df.sum().Y
    # Number of non-churned customers in each bin
    binned_non_null_rows_df["non_churn"] = grouped_tmp_df.count().Y - grouped_tmp_df.sum().Y
    binned_non_null_rows_df = binned_non_null_rows_df.reset_index(drop=True)
    
    if len(null_rows.index) > 0:
        binned_null_rows_df = pd.DataFrame({'min_value': np.nan},index=[0])
        binned_null_rows_df["max_value"] = np.nan
        binned_null_rows_df["count"] = null_rows.count().Y
        binned_null_rows_df["churn"] = null_rows.sum().Y
        binned_null_rows_df["non_churn"] = null_rows.count().Y - null_rows.sum().Y
        binned_rows_df = binned_non_null_rows_df.append(binned_null_rows_df, ignore_index=True)
    else:
        binned_rows_df = binned_non_null_rows_df.copy()
    
    # print(binned_rows_df)
    
    # churn_rate of  a bin = no. of churners of a bin / count of all customers of that bin
    binned_rows_df["churn_rate"] = binned_rows_df['churn'] / binned_rows_df['count']
    binned_rows_df["non_churn_rate"] = binned_rows_df['non_churn'] / binned_rows_df['count']
    # prop_churn (proportion of churners) = no. of churners of a bin / total no. of churners in the sample
    binned_rows_df["prop_churn"] = binned_rows_df['churn'] / binned_rows_df['churn'].sum()
    binned_rows_df["prop_non_churn"] = binned_rows_df['non_churn'] / binned_rows_df['non_churn'].sum()
    binned_rows_df["WOE"] = np.log(binned_rows_df['prop_churn'] / binned_rows_df['prop_non_churn'])
    binned_rows_df["IV"] = (binned_rows_df['prop_churn'] - binned_rows_df['prop_non_churn']) * binned_rows_df["WOE"]
    binned_rows_df["var_name"] = "VAR"
    binned_rows_df = binned_rows_df[['var_name','min_value', 'max_value', 'count', 'churn', 
                                     'churn_rate', 'non_churn', 'non_churn_rate', 'prop_churn',
                                     'prop_non_churn', 'WOE', 'IV']]
    binned_rows_df = binned_rows_df.replace([np.inf, -np.inf], 0)
    binned_rows_df.IV = binned_rows_df.IV.sum()
    
    return binned_rows_df
	
	
def categorical_value_binning(dependent_variable: pd.Series,
                              independent_variable: pd.Series) -> pd.DataFrame:
    """Calculate Weight of Evidence Encoding (WOE) for each category
    and Information Value (IV) for the given independent variable
    
    Arguments:
    ----------
    dependent_variable: pd.Series
        Target variable
    independent_variable: pd.Series
        Independent variable
        
    Returns:
    --------
    binned_rows_df: pd.DataFrame
        Calculated WOE for each category and 
        IV for the given independent variable
    """
    df = pd.DataFrame({"X": independent_variable,
                       "Y": dependent_variable})
    null_rows = df[df['X'].isnull()]
    non_null_rows = df[df['X'].notnull()]    
    grouped_tmp_df = non_null_rows.groupby('X',as_index=True)
    
    binned_non_null_rows_df = pd.DataFrame({},index=[])
    # Count of values in each bin
    binned_non_null_rows_df["count"] = grouped_tmp_df['Y'].count()
    # Min and Max values are same as category
    binned_non_null_rows_df["min_value"] = grouped_tmp_df.sum()['Y'].index
    binned_non_null_rows_df["max_value"] = binned_non_null_rows_df["min_value"]
    # Number of churned customers in each category
    binned_non_null_rows_df["churn"] = grouped_tmp_df.sum()['Y']
    # Number of non-churned customers in each category
    binned_non_null_rows_df["non_churn"] = grouped_tmp_df.count()['Y'] - grouped_tmp_df.sum()['Y']
    
    if len(null_rows.index) > 0:
        binned_null_rows_df = pd.DataFrame({'min_value':np.nan},index=[0])
        binned_null_rows_df["max_value"] = np.nan
        binned_null_rows_df["count"] = null_rows.count()['Y']
        binned_null_rows_df["churn"] = null_rows.sum()['Y']
        binned_null_rows_df["non_churn"] = null_rows.count()['Y'] - null_rows.sum()['Y']
        binned_rows_df = binned_non_null_rows_df.append(binned_null_rows_df, ignore_index=True)
    else:
        binned_rows_df = binned_non_null_rows_df.copy()
    
    binned_rows_df["churn_rate"] = binned_rows_df['churn'] / binned_rows_df['count']
    binned_rows_df["non_churn_rate"] = binned_rows_df['non_churn'] / binned_rows_df['count']
    binned_rows_df["prop_churn"] = binned_rows_df['churn']/binned_rows_df['churn'].sum()
    binned_rows_df["prop_non_churn"] = binned_rows_df['non_churn'] / binned_rows_df['non_churn'].sum()
    binned_rows_df["WOE"] = np.log(binned_rows_df['prop_churn'] / binned_rows_df['prop_non_churn'])
    binned_rows_df["IV"] = (binned_rows_df['prop_churn'] - binned_rows_df['prop_non_churn']) * binned_rows_df['WOE']
    binned_rows_df["var_name"] = "VAR"
    binned_rows_df = binned_rows_df[['var_name','min_value', 'max_value', 'count', 'churn', 
                                     'churn_rate', 'non_churn', 'non_churn_rate', 'prop_churn',
                                     'prop_non_churn', 'WOE', 'IV']]      
    binned_rows_df = binned_rows_df.replace([np.inf, -np.inf], 0)
    binned_rows_df['IV'] = binned_rows_df['IV'].sum()
    binned_rows_df = binned_rows_df.reset_index(drop=True)
    
    return binned_rows_df
	
	
def feature_importance(df: pd.DataFrame,
                       dependent_variable: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gives feature importance for all the columns in
    the 'df'. Below is the predictive power based on the 
    Information Value (IV)
    
    Information Value (IV) | Predictive Power
    -----------------------------------------------
            < 0.02         | Useless for prediction
          0.02 - 0.1       | Weak predictor
           0.1 - 0.3       | Medium predictor
           0.3 - 0.5       | Strong predictor
            > o.5          | Too good predictor
    
    Arguments:
    ----------
    df: pd.DataFrame
        Input dataframe
    dependent_variable: str
        Dependent Variable
        
    Returns:
    --------
    iv_df: pd.DataFrame
        DataFrame with WOE encoded values for each category
        and IV values for each category of the independent variable
    iv: pd.DataFrame
        DataFrame with IV values for each independent variable
        
    Example:
    --------
    >>> iv_df, iv = feature_importance(df, 'Churn_Flag')
    """
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    columns = df.dtypes.index
    count = -1
    
    for column in columns:
        if column.upper() not in (final.upper()):
            if np.issubdtype(df[column], np.number) and len(pd.Series.unique(df[column])) > 2:
                binned_df = continuous_value_binning(dependent_variable, df[column])
                binned_df["var_name"] = column
                count += 1
            else:
                binned_df = categorical_value_binning(dependent_variable, df[column])
                binned_df["var_name"] = column            
                count += 1
                
            if count == 0:
                iv_df = binned_df
            else:
                iv_df = iv_df.append(binned_df, ignore_index=True)

    iv = pd.DataFrame({'IV': iv_df.groupby('var_name')['IV'].max()})
    iv = iv.reset_index()
    return (iv_df, iv)
