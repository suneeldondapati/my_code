from sklearn.tree import DecisionTreeClassifier, export_graphviz
from io import StringIO
import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter


def train_feature_value(X, y_true, feature, value):
    """Selects the most frequent class for the given feature-value pair

    Arguments:
    ----------
    X: pandas.dataframe
        Input dataframe
    feature: str
        IDV
    value: int or float or str
        Value of the corresponding feature
    """
    # Create a simple dictionary to count how frequently 
    # they give certain predictions 
    class_counts = defaultdict(int)
    # Iterate through each sample and count the frequency 
    # of each class/value pair
    for sample, y in zip(X, y_true):
        if sample[feature] == y:
            class_counts[y] += 1
    # Now get the best one by sorting the (highest first) 
    # and choosing the first item
    sorted_class_counts = sorted(class_counts.items(), 
                                 key=itemgetter(1), 
                                 reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    # The error is the number of sample that do not classify
    # as the most frequent class and have the feature value
    n_samples = X.shape[1]
    error = sum([class_count for class_value, class_count in class_counts.items()
                             if class_value != most_frequent_class])
    return most_frequent_class, error


def train(X, y_true, feature):
    # Check that variable is a valid number
    n_samples, n_features = X.shape
    assert 0 <= feature <= n_features
    # Get all the unique values that this variable has
    values = set(X[:, feature])
    # Store the predictors array that is returned
    predictors = dict()
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true,
                                                         feature,
                                                         current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    # Compute the total error of using this feature to classify on
    total_error = sum(errors)
    return predictors, total_error


def rules_from_decision_tree(X, y, tree=False):
    """Get rules for Input data

    Arguments:
    ----------
    X: pandas.Series
        Input Series
    y: pandas.Series
        DV column
    tree: bool
        If True, return tree

    Returns:
    --------
    clf: DecisionTreeClassifier
        Trained classifier
    output_str: str
        Decision rule
    tree: str
        Built tree

    Examples:
    ---------
    >>> tmp = ['age']
    >>> clfs, rules, trees = [], [], []
    >>> for col in tmp:
    ...     clf, rule, tree = rules_from_decision_tree(df[col], 
    ...                                                df['y'], 
    ...                                                tree=True)
    ...     clfs.append(clf);rules.append(rule);trees.append(tree)
    """
    clf = DecisionTreeClassifier(criterion='entropy', 
                                 max_depth=1,
                                 random_state=1310)
    clf.fit(X.values.reshape((-1, 1)), y)
    out = StringIO() 
    export_graphviz(clf, out_file=out, feature_names=[X.name]) 
    feature_decision = re.findall(r'\[label="(.*?)\\n', out.getvalue())
    leaf = re.findall(r'value = \[(.*?)\]', out.getvalue())
    output_str = 'If {} then {}'.format(feature_decision[0],
                                        0 if leaf[1].split(',')[0] > leaf[2].split(',')[0] else 1)
    if tree:
        return clf, output_str, out.getvalue()
    return clf, output_str


