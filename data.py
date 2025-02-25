from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os

class MemoryDataset(Dataset):
    """
    Torch dataset for item response data_pars in numpy array
    """
    def __init__(self, X, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames
        self.x_train = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx]

def parse_arguments(args, cfg):
    # general arguments
    cfg['GeneralConfigs']['repetition'] = args[1]
    cfg['GeneralConfigs']['model'] = args[2]
    cfg['OptimConfigs']['n_iw_samples'] = int(args[3])
    cfg['OptimConfigs']['learning_rate'] = float(args[4])
    cfg['OptimConfigs']['gumbel_decay'] = float(args[5])
    cfg['SimConfigs']['n_items'] = int(args[6])

    # model specifc arguments
    if cfg['GeneralConfigs']['model'] == 'LCA':
        cfg['ModelSpecificConfigs']['n_class'] = int(args[7])
        cfg['ModelSpecificConfigs']['lca_method'] = args[8]
    if cfg['GeneralConfigs']['model'] == 'GDINA':
        cfg['ModelSpecificConfigs']['n_attributes'] = int(args[7])
    if cfg['GeneralConfigs']['model'] == 'MIXIRT':
        cfg['ModelSpecificConfigs']['mirt_dim'] = int(args[7])



    return cfg

def get_paths(cfg):
    if cfg['GeneralConfigs']['model'] == 'LCA':
        arguments = [str(cfg['ModelSpecificConfigs']['n_class']), str(cfg['SimConfigs']['n_items'])]
    if cfg['GeneralConfigs']['model'] == 'GDINA':
        arguments = [str(cfg['ModelSpecificConfigs']['n_attributes']), str(cfg['SimConfigs']['n_items'])]
    if cfg['GeneralConfigs']['model'] == 'MIXIRT':
        arguments = [str(cfg['ModelSpecificConfigs']['mirt_dim'])]
    filename = '_'.join(arguments)

    pars = ['data','class', 'itempars', 'theta']


    base_dir = os.path.abspath('./saved_data')
    model = cfg['GeneralConfigs']['model']
    paths = [os.path.join(base_dir, model, par, f"{filename}") for par in pars]
    return paths


def read_data_pars(cfg):
    """
    Fundtion that loads data, true class membership, true itemparameters and potentially theta from disk
    :param cfg: configurations, detemines which data and parameters to read
    :return: data, class membership, theta and item parameters as np.arrays theta is none of not MIXIRT
    """
    paths = [f'{path}.npy' for path in get_paths(cfg)]

    # for the data we add iteration

    data = np.load(paths[0])
    cl =np.load(paths[1])
    itempars = np.load(paths[2])
    if cfg['GeneralConfigs']['model'] == 'LCA':
        theta = None
        Q = None
    elif cfg['GeneralConfigs']['model'] == 'GDINA':
        theta = None
        Q = (itempars[:, 0:, 1:3] != 0).astype(float)
    elif cfg['GeneralConfigs']['model'] == 'MIXIRT':
        theta = np.load(paths[3])
        Q = (itempars[:, 1:, 0] != 0).astype(float)


    return data, cl, theta, itempars, Q


def write_data_pars(cfg, data, cl, itempars, theta=None):
    """
    Fundtion that writes data, true class membership, true itemparameters and potentially theta to disk
    :param cfg: configurations, detemines where the data and parameters should be saved
    :return: data, class membership, theta and item parameters as np.arrays theta is none of not MIXIRT
    """
    paths = get_paths(cfg) # paths for data, class, itempars and theta respectively
    np.save(paths[0], data)
    np.save(paths[1], cl)
    np.save(paths[2], itempars)
    if theta is not None:
        np.save(paths[3], theta)
    else:
        theta = None

    return data, cl, theta, itempars