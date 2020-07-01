import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_distplots(df, cols=4):
    """
    This function creates distribution plots for all numerical features.
    :param df: dataframe
    :param cols: specified amount of columns in the subplots.
    :return: seaborn distplots
    """
    num_vars = df.select_dtypes('number').columns

    if (len(num_vars) % cols) != 0:
        rows = (len(num_vars) // cols) + 1
    else:
        rows = (len(num_vars) // cols)

    fig, ax = plt.subplots(rows, cols, figsize=(30, 40))
    for variable, subplot in zip(num_vars.tolist(), ax.flatten()):
        sns.distplot(df[variable], ax=subplot)

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, width=50, height=50):
    """
    This function create a correlation heatmap for all numerical features to detect multi-collinearity.
    :param df: dataframe
    :param width: width of the heatmap
    :param height: height of the heatmap
    :return: seaborn heatmap
    """
    fig, ax = plt.subplots(figsize=(width, height))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(df.corr(), cmap=colormap, annot=True)
    plt.show()