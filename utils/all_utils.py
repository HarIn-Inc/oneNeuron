import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
from matplotlib.colors import ListedColormap

plt.style.use('fivethirtyeight')


def prepare_data(df):
    """It is used to separate the dependent variables and independent features
    :param df: DataFrame Object
    :type df:  Pandas dataframe
    :return:  Dataframe object with dependent and independent variables
    :rtype: x1: int64, x2: int64, y: int64
    """
    X = df.drop('y', axis=1)
    y = df['y']

    return X, y


def save_model(model, filename):
    """This saves the trained model
    :param model: Trained model
    :type model: Python Object
    :param filename: Path to save the trained model
    :type filename: str, String
    """
    model_dir = 'Models'
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)


def save_plot(df, file_name, model):
    """
    :param df: Pandas DataFrame
    :type df: Object
    :param file_name: Path to save the plot
    :type file_name: str, String
    :param model: Trained Model
    :type model: obj, Python Object
    """
    def _create_base_plot(df):
        df.plot(kind='scatter', x='x1', y='x2', c='y', s=100, cmap='winter')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        figure = plt.gcf()
        figure.set_size_inches(10, 8)

    def _plot_decision_regions(X, y, classifier, resolution=0.02):
        colors = ('red', 'blue', 'gray', 'cyan', 'beige')
        cmap = ListedColormap(colors[: len(np.unique(y))])

        X = X.values
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        plt.plot()

    X, y = prepare_data(df)

    _create_base_plot(df)
    _plot_decision_regions(X, y, model)

    plot_dir = 'Plots'
    os.makedirs(plot_dir, exist_ok=True)
    plotpath = os.path.join(plot_dir, file_name)
    plt.savefig(plotpath)