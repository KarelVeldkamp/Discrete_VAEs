import sys

import torch
from LCA import *
from GDINA import *
from MIXIRT import *
from helpers import Cor, MSE, recovery_plot, empty_directory
from data import *
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
import matplotlib.pyplot as plt
import shutil

# set working directory to source file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# read in configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['Configs']

# if command line arguments are provided, use these to overwrite configurations
if len(sys.argv) > 1:
    cfg = parse_arguments(sys.argv, cfg)

# simulate data_pars
if cfg['SimConfigs']['sim_data']:
    if cfg['GeneralConfigs']['model'] == 'LCA':
        data, true_class, true_itempars = sim_LCA(
            N=cfg['SimConfigs']['N'],
            nitems=cfg['SimConfigs']['n_items'],
            nclass=cfg['ModelSpecificConfigs']['n_class'],
            sim_pars=cfg['SimConfigs']['sim_pars'])
        true_theta = None
    elif cfg['GeneralConfigs']['model'] == 'GDINA':
        data, true_itempars, true_class = sim_GDINA(
            N=cfg['SimConfigs']['N'],
            nitems=cfg['SimConfigs']['n_items'],
            nattributes=cfg['ModelSpecificConfigs']['n_attributes'],
            sim_pars=cfg['SimConfigs']['sim_pars'])
        Q = torch.Tensor(true_itempars[:, 0,1:] != 0).float()
        true_theta = None
    elif cfg['GeneralConfigs']['model'] == 'MIXIRT':
        if cfg['ModelSpecificConfigs']['mirt_dim'] > 1:
            Q = pd.read_csv(f'./Qmatrices/QMatrix{cfg["ModelSpecificConfigs"]["mirt_dim"]}D.csv', header=None).values.astype(float)
        else:
            Q = np.ones((cfg['SimConfigs']['n_items'], cfg['ModelSpecificConfigs']['mirt_dim']))

        data, true_class, true_theta, true_itempars = sim_MIXIRT(
            N=cfg['SimConfigs']['N'],
            nitems=cfg['SimConfigs']['n_items'],
            nclass=2,
            mirt_dim=cfg['ModelSpecificConfigs']['mirt_dim'],
            Q = Q,
            class_prob=cfg['ModelSpecificConfigs']['class_prob'],
            cov=cfg['ModelSpecificConfigs']['cov'],
            sim_pars=cfg['SimConfigs']['sim_pars'])

    # potentially save simualted data and parametes to disk
    if cfg['SimConfigs']['save_data_pars']:
        write_data_pars(cfg, data, true_class, true_itempars, true_theta)
        exit()
# or read data_pars from disk
else:
    data, true_class, true_theta, true_itempars, Q = read_data_pars(cfg)


#true_itempars = torch.Tensor(true_itempars)
# intiralize data_pars loader
dataset = MemoryDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['OptimConfigs']['batch_size'], shuffle=True)
test_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)

# empty logs
#empty_directory('./logs/')

# repeat the enitre training process n_rep times
best_ll = -float('inf')
for i in range(cfg['OptimConfigs']['n_rep']):
    # initialize logger and trainer
    logger = CSVLogger("logs", name='_'.join(sys.argv), version=0)
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
                  min_temp=cfg['OptimConfigs']['gumbel_min_temp'],
                  beta=1,
                  nclass=2)

    start = time.time()
    trainer.fit(model)
    runtime = time.time() - start
    print(f'runtime: {runtime}')
    #print(f'final temperature: {model.sampler.temperature}')

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
    lc_acc = (true_class== (pi > .5).float().detach().numpy()).mean()

if cfg['GeneralConfigs']['model'] == 'MIXIRT':
    # The latent dimension in the mixture is only identified up to the sign so we might have to flip the sign:
    for dim in range(theta.shape[1]):
        if pearsonr(true_theta[:, dim], theta[:, dim])[0] < 0:
            best_theta[:, dim] *= -1
            best_itempars[:,dim+1, :] *= -1


    mse_theta = MSE(best_theta.detach().numpy(), true_theta)
    bias_theta = np.mean(best_theta.detach().numpy() - true_theta)
    var_theta = np.var(best_theta.detach().numpy())
else:
    mse_theta = bias_theta = var_theta = None





# compute MSE of conditional probabilities
mse_itempars = MSE(best_itempars.detach().numpy()[true_itempars!=0], true_itempars[true_itempars!=0])

bias_itempars = np.mean(best_itempars.detach().numpy()[true_itempars!=0] - true_itempars[true_itempars!=0])
var_itempars = np.var(best_itempars.detach().numpy()[true_itempars!=0])

# save MSE for a and b separately in the MIXIRt model
if cfg['GeneralConfigs']['model'] == 'MIXIRT':
    best_a = best_itempars[:,0,:]
    true_a = true_itempars[:,0,:]
    best_b = best_itempars[:,1:,:]
    true_b = true_itempars[:, 1:, :]
    msea = MSE(best_a.detach().numpy()[true_a!=0], true_a[true_a!=0])
    biasa = np.mean(best_a.detach().numpy()[true_a!=0]- true_a[true_a!=0])
    mseb = MSE(best_b.detach().numpy(), true_b)
    biasb = np.mean(best_b.detach().numpy()-true_b)
else:
    msea = biasa = mseb = biasb = None


# plotting
if cfg['GeneralConfigs']['save_plot']:
    empty_directory('./figures/')
    # save recovery plot of item parameters
    recovery_plot(true=true_itempars[true_itempars!=0],
                  est=best_itempars.detach().numpy()[true_itempars!=0],
                  name='Overall_Itemparameter_recovery')
    # save recovery plot of ability parameters for the MIXIRT model
    if cfg['GeneralConfigs']['model'] == 'MIXIRT':
        recovery_plot(true=true_theta,
                      est=best_theta.detach().numpy(),
                      name='Overall_Theta_recovery')
    if cfg['GeneralConfigs']['separate_plots']:
        for cl in range(true_itempars.shape[2]):
            for dim in range(true_itempars.shape[1]):

                best_itempars_dim = best_itempars[:,dim, cl]
                true_itempars_dim = true_itempars[:,dim, cl]

                recovery_plot(true=true_itempars_dim[true_itempars_dim!= 0],
                              est=best_itempars_dim.detach().numpy()[true_itempars_dim!= 0],
                              name=f'Itemparameter_recovery_class{cl+1}_dim{dim+1}')

    # read the logs and remove them after (we only save the plot)
    logs = pd.read_csv(f'logs/{"_".join(sys.argv)}/version_0/metrics.csv')
    shutil.rmtree(f'logs/{"_".join(sys.argv)}/')

    plt.figure()
    plt.plot(logs['epoch'], logs['train_loss'])
    plt.title('Training loss')
    plt.savefig(f'./figures/training_loss.png')
if cfg['GeneralConfigs']['save_parameter_estimates']:
    par_names = ['class', 'itempars', 'theta']

    par = []
    value = []
    par_i = []
    par_j = []
    for i, est in enumerate([pi, itempars, theta]):
        if est is not None:
            for r in range(est.shape[0]):
                for c in range(est.shape[1]):
                    par.append(par_names[i])
                    value.append(est[r, c].detach().numpy())
                    par_i.append(r)
                    par_j.append(c)

    result = pd.DataFrame({'parameter': par, 'i': par_i, 'j': par_j, 'value': value})
    result.to_csv(f"./results/estimates/estimates_{'_'.join(sys.argv[1:])}.csv", index=False)
if cfg['GeneralConfigs']['save_metrics']:
    metrics = [lc_acc, mse_itempars, mse_theta, var_itempars, var_theta, bias_itempars, bias_theta, runtime, mseb, msea, biasb, biasa]
    with open(f"./results/metrics/metrics_{'_'.join(sys.argv[1:])}.csv", 'w') as f:
        for metric in metrics:
            f.write(f"{metric}\n")

