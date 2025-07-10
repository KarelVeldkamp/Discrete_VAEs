import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
def MSE(est, true):
    """
    Mean square error
    Parameters
    ----------
    est: estimated parameter values
    true: true paremters values

    Returns
    -------
    the MSE
    """
    return np.mean(np.power(est-true,2))


def Cor(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def recovery_plot(true, est, name):
    """
    create a scatterplot plotting parameter estimates against the true values
    :param true: array of true parameters
    :param est: array of estimates
    :param name: name to use in the plot file
    :return: None
    """
    # flatten probs for plotting

    mse = MSE(est.flatten(), true.flatten())

    plt.figure()
    plt.scatter(y=est.flatten(), x=true.flatten())
    plt.plot(true.flatten(), true.flatten())
    plt.title(f'Probability estimation plot:, MSE={round(mse, 4)}')
    plt.xlabel('True values')
    plt.ylabel('Estimates')
    plt.savefig(f'../figures/{name}.png')

    return None


def empty_directory(dir_path):
    # Recursively remove all contents of the directory
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directory and its contents
        else:
            os.remove(item_path)  # Remove file