import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import seaborn as sns
import numpy as np
import pandas as pd

MARKERS = ['o','v','^','s','p','*','X','H','D']

def bars_categories_count(df, col, title=None):
    data = df.groupby(col).count()
    #index = data.columns.tolist().index(col)
    #counts = data.iloc[:,-1] if index != len(data.columns)-1 else data.iloc[:,-2]
    fig, ax = plt.subplots()
    with sns.axes_style("darkgrid"):
        ax = sns.barplot(x=data.index.tolist(), y=data.max(axis=1), ax=ax)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
        if title:
            ax.set_title(title)
    return fig, ax

def bar_plot_series(series, title=None, palette=None):
    fig, ax = plt.subplots()
    with sns.axes_style("darkgrid"):
        ax = sns.barplot(x=series.index.tolist(), y=series.values, ax=ax, palette=palette)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    if title:
        ax.set_title(title)
    return fig, ax

def bar_plot_binary_features(df, col_names=None, title=None):
    df_features = df.loc[:,col_names] if col_names else df.loc[:,:]
    totals = df_features.sum(axis=0)
    fig, ax = bar_plot_series(totals, title)
    return fig, ax

def correlation_matrix(array, feature_names=None):
    """If array is a numpy array, necessary to pass feature_names.
    If array is a pandas DataFrame, feature_names is not used"""
    if type(array) == np.ndarray:
        sns.pairplot(pd.DataFrame(array, columns=feature_names))
    elif type(array) == pd.DataFrame:
        sns.pairplot(array)

def plot_clusters_2d(groups_assigned, reduced_data, labels=None, title=None,
                    legend_title=None, colormap=None, xmin=None, xmax=None, ymin=None, ymax=None):
    """In the workflow (data_wrangling.py) the groups are found under the dict true_groups
    and the labels under unique_labels"""
    cmap = plt.get_cmap('tab20') if not colormap else plt.get_cmap(colormap)
    colorlist = cmap.colors[:len(groups_assigned)]
    if labels is None:
        labels = list(groups_assigned.keys())
    fig, ax = plt.subplots()
    for k, v in groups_assigned.items():
        points = reduced_data[v['samples_from_zero']]
        if xmin or xmax or ymin or ymax:
            points = clean_outliers(points, xmin, xmax, ymin, ymax)
        ax.scatter(points[:,0], points[:,1], label=labels[k], color=colorlist[k], marker=MARKERS[k], alpha=0.4)
        if not legend_title:
            ax.legend(title='Cluster id')
        else:
            ax.legend(title=legend_title)
    if title:
        ax.set_title(title)
    return fig, ax

def plot_clusters_3d(groups_assigned, reduced_data, labels=None, title=None,
                    legend_title=None, colormap=None, xmin=None, xmax=None, ymin=None, ymax=None):
    """In the workflow (data_wrangling.py) the groups are found under the dict true_groups
    and the labels under unique_labels"""
    cmap = plt.get_cmap('tab20') if not colormap else plt.get_cmap(colormap)
    colorlist = cmap.colors[:len(groups_assigned)]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    if labels is None:
        labels = list(groups_assigned.keys())

    for k, v in groups_assigned.items():
        points = reduced_data[v['samples_from_zero']]
        if xmin or xmax or ymin or ymax:
            points = clean_outliers(points, xmin, xmax, ymin, ymax)
        ax.scatter(points[:,0], points[:,1], points[:,2], label=labels[k], color=colorlist[k], marker=MARKERS[k], alpha=0.4)
        if not legend_title:
            ax.legend(title='Cluster id')
        else:
            #ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))
            f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
            ax.legend(title=legend_title, loc="lower right", bbox_to_anchor=f(-35,30,-25), 
                        bbox_transform=ax.transData)
    if title:
        ax.set_title(title)
    return fig, ax

def clean_outliers(points, xmin, xmax, ymin, ymax):
    points = points[np.where(points[:,0] > xmin)] if xmin else points
    points = points[np.where(points[:,1] > ymin)] if ymin else points
    points = points[np.where(points[:,0] < xmax)] if xmax else points
    points = points[np.where(points[:,1] < ymax)] if ymax else points
    return points