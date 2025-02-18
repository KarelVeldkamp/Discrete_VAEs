import torch

from LCA import *
from GDINA import *
from MIXIRT import *
from helpers import Cor, MSE, recovery_plot, empty_directory
from data import MemoryDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.optimize import linear_sum_assignment
from pytorch_lightning.loggers import CSVLogger
import yaml
import time
import os
import pandas as pd
from scipy.stats import pearsonr

# read in configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['Configs']

# simulate data
if cfg['GeneralConfigs']['simdata']:
    if cfg['GeneralConfigs']['model'] == 'LCA':
        data, true_class, true_itempars = sim_LCA(
            N=cfg['SimConfigs']['N'],
            nitems=cfg['SimConfigs']['n_items'],
            nclass=cfg['ModelSpecificConfigs']['n_class'])
    elif cfg['GeneralConfigs']['model'] == 'GDINA':
        data, true_itempars, true_class, true_eff = sim_GDINA(
            N=cfg['SimConfigs']['N'],
            nitems=cfg['SimConfigs']['n_items'],
            nattributes=cfg['ModelSpecificConfigs']['n_attributes'])
        Q = torch.Tensor(true_itempars[:, 0,1:] != 0).float()

    elif cfg['GeneralConfigs']['model'] == 'MIXIRT':
        if cfg['ModelSpecificConfigs']['mirt_dim'] > 1:
            Q = pd.read_csv(f'./QMatrices/QMatrix{cfg['ModelSpecificConfigs']['mirt_dim']}DSimple.csv', header=None).values.astype(float)
        else:
            Q = np.ones((cfg['SimConfigs']['n_items'], cfg['ModelSpecificConfigs']['mirt_dim']))

        data, true_class, true_theta, true_itempars = sim_MIXIRT(
            N=cfg['SimConfigs']['N'],
            nitems=cfg['SimConfigs']['n_items'],
            nclass=2,
            mirt_dim=cfg['ModelSpecificConfigs']['mirt_dim'],
            Q = Q,
            class_prob=cfg['ModelSpecificConfigs']['class_prob'],
            cov=cfg['ModelSpecificConfigs']['cov'])
# or read data from disk
else:
    true_class = pd.read_csv(f'~/Documents/GitHub/VAE_MIXIRT/true/pars/class_{3}_{0.3}.csv').values.astype('int')
    true_theta = pd.read_csv(f'~/Documents/GitHub/VAE_MIXIRT/true/pars/theta_{3}_{0.3}.csv').values.astype('float')
    true_difficulty = pd.read_csv(f'~/Documents/GitHub/VAE_MIXIRT/true/pars/difficulty_{3}_{0.3}.csv').values.astype('float')
    true_slopes0 = pd.read_csv(f'~/Documents/GitHub/VAE_MIXIRT/true/pars/slopes0_{3}_{0.3}.csv').values.astype('float')
    true_slopes1 = pd.read_csv(f'~/Documents/GitHub/VAE_MIXIRT/true/pars/slopes1_{3}_{0.3}.csv').values.astype('float')
    true_slopes = np.concatenate((np.expand_dims(true_slopes0, -1), np.expand_dims(true_slopes1, -1)),
                                 -1)  # repeat for both classes
    data = pd.read_csv(f'~/Documents/GitHub/VAE_MIXIRT/true/data/data_{3}_{0.3}_{1}.csv').values.astype('float')

    Q = true_slopes[:, :, 0] != 0

    true_itempars = np.concatenate((true_difficulty[:, np.newaxis, :], true_slopes), 1)



#true_itempars = torch.Tensor(true_itempars)
# intiralize data loader
dataset = MemoryDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['OptimConfigs']['batch_size'], shuffle=True)
test_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)

# empty logs
empty_directory('./logs/')



# repeat the enitre training process n_rep times
best_ll = -float('inf')
for i in range(cfg['OptimConfigs']['n_rep']):
    # initialize logger and trainer
    logger = CSVLogger("logs", name='all', version=0)
    trainer = Trainer(fast_dev_run=cfg['OptimConfigs']['single_epoch_test_run'],
                      max_epochs=cfg['OptimConfigs']['max_epochs'],
                      min_epochs=cfg['OptimConfigs']['min_epochs'],
                      logger=logger,
                      callbacks=[
                          EarlyStopping(monitor='train_loss',
                                        min_delta=cfg['OptimConfigs']['min_delta'],
                                        patience=cfg['OptimConfigs']['patience'],
                                        mode='min')],
                      enable_progress_bar=True,
                      enable_model_summary=False,
                      detect_anomaly=cfg['OptimConfigs']['detect_anomaly'],
                      accelerator=cfg['OptimConfigs']['accelerator'])

    # fit the model (LCA, GDINA or MIXIRT)
    if cfg['GeneralConfigs']['model'] == 'LCA':
        if cfg['ModelSpecificConfigs']['lca_method'] in ['dvae', 'gs', 'vq', 'log']:
            model = LCA(dataloader=train_loader,
                        nitems=data.shape[1],
                        nclass=cfg['ModelSpecificConfigs']['n_class'],
                        hidden_layer_size=(data.shape[1]+cfg['ModelSpecificConfigs']['n_class'])//2,
                        learning_rate=cfg['OptimConfigs']['learning_rate'],
                        emb_dim=cfg['ModelSpecificConfigs']['emb_dim'],
                        temperature=cfg['OptimConfigs']['gumbel_temperature'],
                        temperature_decay=cfg['OptimConfigs']['gumbel_decay'],
                        sampler_type=cfg['ModelSpecificConfigs']['lca_method'],
                        min_temp=cfg['OptimConfigs']['gumbel_min_temp'],
                        n_iw_samples=cfg['OptimConfigs']['n_iw_samples'],
                        beta=1)
        elif cfg['ModelSpecificConfigs']['lca_method'] == 'rbm':
            nnodes = np.log2(cfg['ModelSpecificConfigs']['n_class'])
            if not nnodes.is_integer():
                raise ValueError('RBM only implemented for 2, 4, 8, ... classes')
            model = RestrictedBoltzmannMachine(
                dataloader=train_loader,
                n_visible=data.shape[1],
                n_hidden=int(nnodes),
                learning_rate=cfg['OptimConfigs']['learning_rate'],
                n_gibbs=cfg['ModelSpecificConfigs']['gibbs_samples']
            )
        else:
            raise ValueError('Invalid model type')

    elif cfg['GeneralConfigs']['model'] == 'GDINA':
        model = GDINA(dataloader=train_loader,
                      n_items=data.shape[1],
                      n_attributes=cfg['ModelSpecificConfigs']['n_attributes'],
                      Q=Q,
                      learning_rate=cfg['OptimConfigs']['learning_rate'],
                      temperature=cfg['OptimConfigs']['gumbel_temperature'],
                      temperature_decay=cfg['OptimConfigs']['gumbel_decay'],
                      min_temp=cfg['OptimConfigs']['gumbel_min_temp'],
                      n_iw_samples=cfg['OptimConfigs']['n_iw_samples']
                      )

    elif cfg['GeneralConfigs']['model'] == 'MIXIRT':
        model = VAE(dataloader=train_loader,
                  nitems=data.shape[1],
                  learning_rate=cfg['OptimConfigs']['learning_rate'],
                  latent_dims=cfg['ModelSpecificConfigs']['mirt_dim'],
                  hidden_layer_size=50,
                  qm=Q,
                  batch_size=cfg['OptimConfigs']['batch_size'],
                  n_iw_samples=cfg['OptimConfigs']['n_iw_samples'],
                  temperature=cfg['OptimConfigs']['gumbel_temperature'],
                  temperature_decay=cfg['OptimConfigs']['gumbel_decay'],
                  beta=1)

    start = time.time()
    trainer.fit(model)
    runtime = time.time() - start
    print(f'runtime: {runtime}')

    # check if the model fit is better than previous repetitions
    pi, theta, itempars, ll = model.compute_parameters(data)

    if ll > best_ll:
        best_ll = ll
        best_model = model
        best_itempars = itempars
        best_pi = pi
        best_theta = theta
        best_class_ix = torch.argmax(best_pi, 1)

# for mixture IRT and LCA we have to account for label switching:
if  cfg['GeneralConfigs']['model'] in ['MIXIRT', 'LCA']:

    # print(Cor(best_itempars.detach().numpy(), true_itempars.detach().numpy()))
    #
    # cor = Cor(best_itempars.detach().numpy(), true_itempars.detach().numpy())
    # #cor = Cor(best_itempars.flatten(0,1).detach().numpy().T, true_itempars.flatten(0,1).detach().numpy().T)
    # #cor = Cor(best_itempars[:,0,:].detach().numpy().T, true_itempars[:,0,:].detach().numpy().T)
    #
    # cor[np.isnan(cor)] = 1
    # _, new_order = linear_sum_assignment(-cor)
    # true_itempars = true_itempars[new_order, :]
    # true_class = true_class[:, new_order]


    _, new_order = linear_sum_assignment(-Cor(best_itempars.flatten(0,1).detach().numpy().T,
                                              torch.Tensor(true_itempars).flatten(0,1).detach().numpy().T
                                              )
                                         )

    true_itempars = true_itempars[:,:, new_order]

    true_class = true_class[:, new_order]
    true_class_ix = np.argmax(true_class, 1)
    # compute latent class accuracy
    lc_acc = np.mean(best_class_ix.detach().numpy() == true_class_ix)
elif cfg['GeneralConfigs']['model'] == 'GDINA':

    lc_acc = (true_class.detach().numpy() == (pi > .5).float().detach().numpy()).mean()
# The latent dimension in the mixture is only identified up to the sign so we might have to flip the sign:
if cfg['GeneralConfigs']['model'] == 'MIXIRT':
    for dim in range(theta.shape[1]):
        print(pearsonr(true_theta[:, dim], theta[:, dim]).statistic)
        if pearsonr(true_theta[:, dim], theta[:, dim]).statistic < 0:
            best_theta[:, dim] *= -1
            best_itempars[:,dim+1, :] *= -1



# print(true_class.shape)
# print(best_pi.shape)
# cor = Cor(true_class.T, best_pi.T.detach().numpy())
# cor[np.isnan(cor)] = 1
# _, new_order = linear_sum_assignment(-cor)



# compute MSE of conditional probabilities
mse_cond = MSE(best_itempars.detach().numpy(), true_itempars)


print(lc_acc)
print(mse_cond)


# plotting
empty_directory('./figures/')
if cfg['GeneralConfigs']['save_plot']:
    # save recovery plot of item parameters in /figures/
    recovery_plot(true=true_itempars[true_itempars!=0],
                  est=best_itempars.detach().numpy()[true_itempars!=0],
                  name='Overall_Itemparameter_recovery')

    if cfg['GeneralConfigs']['separate_plots']:
        for cl in range(true_itempars.shape[2]):
            for dim in range(true_itempars.shape[1]):

                best_itempars_dim = best_itempars[:,dim, cl]
                true_itempars_dim = true_itempars[:,dim, cl]

                recovery_plot(true=true_itempars_dim[true_itempars_dim!= 0],
                              est=best_itempars_dim.detach().numpy()[true_itempars_dim!= 0],
                              name=f'Itemparameter_recovery_ls{cl+1}_dim{dim+1}')





