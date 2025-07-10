import sys
from LCA import *
from GDINA import *
from MIXIRT import *
from data import *
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import yaml
import time
import pandas as pd


# read in configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['Configs']

# Read command line arguments
if len(sys.argv) > 1:
    model_type = sys.argv[1]
    data_path = sys.argv[2]
    nclass = int(sys.argv[3])
    if model_type != 'LCA':
        Q_path = sys.argv[4]


data = pd.read_csv(data_path, header=None).values
print(data.shape)
if model_type != 'LCA':
    Q = pd.read_csv(Q_path, header=None).values
    if model_type == 'GDINA':
        Q = expand_interactions(torch.Tensor(Q)).squeeze()


dataset = MemoryDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['OptimConfigs']['batch_size'], shuffle=True)
test_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)

# empty logs
# empty_directory('./logs/')

# repeat the enitre training process n_rep times
best_ll = -float('inf')
for i in range(cfg['OptimConfigs']['n_rep']):
    print(f'starting rep {i+1}/{cfg['OptimConfigs']['n_rep']}')
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
    if model_type == 'LCA':
        model = LCA(dataloader=train_loader,
                    nitems=data.shape[1],
                    nclass=nclass,
                    hidden_layer_size=(data.shape[1] + cfg['ModelSpecificConfigs']['n_class']) // 2,
                    learning_rate=cfg['OptimConfigs']['learning_rate'],
                    emb_dim=cfg['ModelSpecificConfigs']['emb_dim'],
                    temperature=cfg['OptimConfigs']['gumbel_temperature'],
                    temperature_decay=cfg['OptimConfigs']['gumbel_decay'],
                    sampler_type=cfg['ModelSpecificConfigs']['lca_method'],
                    min_temp=cfg['OptimConfigs']['gumbel_min_temp'],
                    n_iw_samples=cfg['OptimConfigs']['n_iw_samples'],
                    beta=1)
    elif model_type == 'GDINA':
        model = GDINA(dataloader=train_loader,
                      n_items=data.shape[1],
                      n_attributes=nclass,
                      Q=Q,
                      learning_rate=cfg['OptimConfigs']['learning_rate'],
                      temperature=cfg['OptimConfigs']['gumbel_temperature'],
                      temperature_decay=cfg['OptimConfigs']['gumbel_decay'],
                      min_temp=cfg['OptimConfigs']['gumbel_min_temp'],
                      n_iw_samples=cfg['OptimConfigs']['n_iw_samples']
                      )
    elif model_type == 'MIXIRT':
        model = VAE(dataloader=train_loader,
                    nitems=data.shape[1],
                    learning_rate=cfg['OptimConfigs']['learning_rate'],
                    latent_dims=Q.shape[1],
                    hidden_layer_size=50,
                    qm=Q,
                    batch_size=cfg['OptimConfigs']['batch_size'],
                    n_iw_samples=cfg['OptimConfigs']['n_iw_samples'],
                    temperature=cfg['OptimConfigs']['gumbel_temperature'],
                    temperature_decay=cfg['OptimConfigs']['gumbel_decay'],
                    min_temp=cfg['OptimConfigs']['gumbel_min_temp'],
                    beta=1,
                    nclass=nclass)

    start = time.time()
    trainer.fit(model)
    runtime = time.time() - start


    print(f'runtime rep {i+1}/{cfg['OptimConfigs']['n_rep']}: {runtime}')
    #print(f'final temperature: {model.sampler.temperature}')

    # check if the model fit is better than previous repetitions
    print('computing EAP estimates...')
    start = time.time()
    pi, theta, itempars, ll = model.compute_parameters(data)
    runtime = time.time() - start
    print(f'runtime EAP estimates {i + 1}: {runtime}')
    print(f'log-likelihood rep {i+1}/{cfg['OptimConfigs']['n_rep']}: {ll}')

    if isinstance(itempars, torch.Tensor):
        itempars = itempars.detach().numpy()

    if ll > best_ll:
        best_rep = i
        best_ll = ll
        best_model = model
        best_itempars = itempars
        best_pi = pi
        best_theta = theta
        best_class_ix = torch.argmax(best_pi, 1)

print(f'Best model: model {best_rep+1}, log-likelihood: {best_ll}')
print('Saving parameter estimates to ./results/estimates/')

np.savetxt('./results/estimates/class_probabilities.csv', best_pi, delimiter=',')
if model_type == 'MIXIRT':
    np.savetxt('./results/estimates/abilities.csv', best_theta, delimiter=',')
if model_type == 'GDINA':
    # write a list of understandable column names for the GDINA model (to show which column corresponds to which effect)
    n_effects = 2 ** nclass - 1
    required_mask = torch.arange(1, n_effects + 1).unsqueeze(1).bitwise_and(1 << torch.arange(nclass)).bool()
    colnames = ['int']
    for row in required_mask:
        active = [str(i + 1) for i, val in enumerate(row) if val == 1.0]
        colnames.append('-'.join(active))
    header = ','.join(colnames)
else:
    header = None

if model_type!='MIXIRT':
    np.savetxt('./results/estimates/item_parameters.csv', best_itempars.squeeze(), delimiter=',', header=header)
else:
    for cl in range(best_itempars.shape[2]):
        np.savetxt(f'./results/estimates/item_parameters_class{cl+1}.csv', best_itempars[:, :, cl], delimiter=',')




