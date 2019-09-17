#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Filename: other_plots
Date: 2019-05-10 10:55
Project: AXA
AUTHOR: Suneel Dondapati
"""


import numpy as np
from scipy.cluster.hierarchy import dendrogram

__all__ = [
    'plot_dendrogram',
]


def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

