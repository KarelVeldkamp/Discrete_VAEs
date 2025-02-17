import torch

from LCA import *
from GDINA import *
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


# read in configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['Configs']

# simulate data
if cfg['GeneralConfigs']['simdata']:
    if cfg['GeneralConfigs']['model'] == 'LCA':
        data, true_class, true_itempars = sim_LCA(N=cfg['SimConfigs']['N'],
                                               nitems=cfg['SimConfigs']['n_items'],
                                               nclass=cfg['ModelSpecificConfigs']['n_class'])
    if cfg['GeneralConfigs']['model'] == 'GDINA':
        data, true_itempars, true_class, true_eff = sim_GDINA(N=cfg['SimConfigs']['N'],
                                          nitems=cfg['SimConfigs']['n_items'],
                                          nattributes=cfg['ModelSpecificConfigs']['n_attributes'])
        Q = torch.Tensor(true_itempars[:,1:] != 0).float()



# or read data from disk
else:
    raise NotImplementedError()

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
                      detect_anomaly=True,
                      accelerator=cfg['OptimConfigs']['accelerator'])
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

    start = time.time()
    trainer.fit(model)
    runtime = time.time() - start
    print(f'runtime: {runtime}')

    # check if the model fit is better than previous repetitions
    pi, itempars, ll = model.compute_parameters(data)

    if ll > best_ll:
        best_ll = ll
        best_model = model
        best_itempars = itempars
        best_pi = pi
        best_class_ix = torch.argmax(best_pi, 1)


# match estimated latent classes to the correct true class


_, new_order = linear_sum_assignment(-Cor(best_itempars.detach().numpy(), true_itempars))
true_itempars = true_itempars[new_order, :]
true_class = true_class[:, new_order]
true_class_ix = np.argmax(true_class, 1)


# compute latent class accuracy
lc_acc = np.mean(best_class_ix.detach().numpy() == true_class_ix)
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
        for dim in range(true_itempars.shape[0]):
            best_itempars_dim = best_itempars[dim,:]
            true_itempars_dim = true_itempars[dim,:]
            recovery_plot(true=true_itempars_dim[true_itempars_dim != 0],
                          est=best_itempars_dim.detach().numpy()[true_itempars_dim != 0],
                          name=f'Itemparameter_recovery_{dim+1}')





