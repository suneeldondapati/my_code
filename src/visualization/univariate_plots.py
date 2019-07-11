from matplotlib.ticker import PercentFormatter
from scipy.stats import skew, kurtosis
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd


__all__ = [
    'univariate_plot',
    'univariate_plot_v2',
    'boxplot_summary',
    'pareto_plot',
]


def univariate_plot_v2(independent_variable: pd.Series, 
                       dependent_variable: pd.Series):
    """
    Plots histogram and KDE for continous variable
    
    Aguments:
    ---------
    independent_variable: pd.Series
        Independent feature column
    dependent_variable: pd.Series
        Target column
    
    Returns:
    --------
    Plots univariate subplots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter
    
    df = pd.concat([independent_variable, dependent_variable], axis=1)
    iv_name = independent_variable.name
    dv_name = dependent_variable.name
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey=False, figsize=(12, 8))
    ax1.hist(df[iv_name].values.reshape(-1, 1), bins=50)
    sns.kdeplot(df[iv_name].dropna(), color='navy', ax=ax2)
    grouped_df = df.groupby(dv_name)
    non_churned_vals = grouped_df.get_group(0)[iv_name].values.reshape(-1, 1)
    churned_vals = grouped_df.get_group(1)[iv_name].values.reshape(-1, 1)
    sns.kdeplot(df[df[dv_name] == 0][iv_name].dropna(), color='navy', label='No Churn', ax=ax3)
    sns.kdeplot(df[df[dv_name] == 1][iv_name].dropna(), color='orange', label='Churn', ax=ax3)
    ax4.hist(non_churned_vals, bins=50, alpha=0.3, color='blue')
    ax4.yaxis.set_major_formatter(PercentFormatter(xmax=len(non_churned_vals)))
    ax4.set_ylabel('% non-churned', color='blue')
    ax5 = ax4.twinx()
    ax5.hist(churned_vals, bins=50, alpha=0.3, color='red')
    ax5.yaxis.set_major_formatter(PercentFormatter(xmax=len(churned_vals)))
    ax5.set_ylabel('% churned', color='red')
    fig.suptitle(f'{iv_name}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()
    
def univariate_plot(df, feature):
    """
    Plots histogram and KDE for continous variable
    
    Aguments:
    ---------
    df: pd.DataFrame
        Input data frame
    feature: str
        Feature on which univariate analysis has to performed
    
    Returns:
    --------
    Plots univariate subplots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey=False, figsize=(12, 8))
    ax1.hist(df[feature].values.reshape(-1, 1), bins=50)
    sns.kdeplot(df[feature].dropna(), color='navy', ax=ax2)
    grouped_df = df.groupby('Churn_Flag')
    non_churned_vals = grouped_df.get_group(0)[feature].values.reshape(-1, 1)
    churned_vals = grouped_df.get_group(1)[feature].values.reshape(-1, 1)
    sns.kdeplot(df[df['Churn_Flag'] == 0][feature].dropna(), color='navy', label='No Churn', ax=ax3)
    sns.kdeplot(df[df['Churn_Flag'] == 1][feature].dropna(), color='orange', label='Churn', ax=ax3)
    ax4.hist(non_churned_vals, bins=50, alpha=0.3, color='blue')
    ax4.yaxis.set_major_formatter(PercentFormatter(xmax=len(non_churned_vals)))
    ax4.set_ylabel('% non-churned', color='blue')
    ax5 = ax4.twinx()
    ax5.hist(churned_vals, bins=50, alpha=0.3, color='red')
    ax5.yaxis.set_major_formatter(PercentFormatter(xmax=len(churned_vals)))
    ax5.set_ylabel('% churned', color='red')
    fig.suptitle(f'{feature}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()


def boxplot_summary(column: pd.Series):
    """Get summary stats of box-plot
    
    Arguments:
    ----------
    column: pd.Series
        Input column for summary
        
    Returns:
    --------
    Summary of the box-plot
    
    Example:
    --------
    >>> boxplot_summary(df['column'])

    Reference:
    ----------
    Skewness: Skewness for normal distribution is zero.
              Any symmetric data should have a skewness near to zero.
              Negative values indicate, data is skewed left.
              Positive values indicate, data is skewed right.

    Kurtosis: Kurtosis for standard normal distribution is
              0, if fisher's definition is used.
              3, if pearson's  definition is used.
              With respect to fisher's definition
              Positive kurtosis indicate heavy tailed
              Negative kurtosis indicate light tailed
    """
    col_desc = column.describe()
    q1 = col_desc.loc['25%']
    q3 = col_desc.loc['75%']
    iqr = q3 - q1
    lower_extreme = q1 - 1.5*iqr
    upper_extreme = q3 + 1.5*iqr
    col_desc.at['lower-extreme'] = lower_extreme
    col_desc.at['upper-extreme'] = upper_extreme
    col_desc.at['median'] = column.median()
    col_desc.at['iqr'] = iqr
    col_desc.at['skewness'] = skew(column)
    col_desc.at['kurtosis'] = kurtosis(column)
    print(col_desc)


def pareto_plot(column: pd.Series,
                use_given_index: bool = False,
                figsize: Tuple[int, int] = (12, 8),
                return_freq_df: bool = False):
    """
    Draw Pareto plot for categorical variable

    Arguments:
    ----------
    column: pd.Series
        Categorical input
    figsize: Tuple
        size of the figure
    return_freq_df: bool
        Returns frequency dataframe if True

    Example:
    --------
    >>> pareto_plot(df['state'], figsize=(20, 10))

    >>> df = pareto_plot(df['area code'], return_freq_df=True)
    >>> df
       label  frequency  cumpercentage
    0    415       1655      49.654965
    1    510        840      74.857486
    2    408        838     100.000000
    """
    freq = column.copy()
    if use_given_index:
        freq = column.value_counts().sort_values(ascending=False)
    freq_df = pd.DataFrame({'label': freq.index,
                            'frequency': freq.values})
    freq_df['cumpercentage'] = freq_df['frequency'].cumsum()/freq_df['frequency'].sum()*100
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(freq_df.index, freq_df['frequency'],
           color='C0')
    ax2 = ax.twinx()
    ax2.plot(freq_df.index, freq_df['cumpercentage'],
             color='C1', marker='D', ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xticks(freq_df.index)
    ax.set_xticklabels(freq_df['label'], fontsize=10,
                       rotation=35)
    ax.tick_params(axis='y', colors='C0')
    ax2.tick_params(axis='y', colors='C1')
    plt.show()
    if return_freq_df:
        return freq_df
