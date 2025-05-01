import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
import torch.distributions as dist
from samplers import GumbelSampler, VectorQuantizer, LogisticSampler, StraightThroughSampler, SpikeAndExp
import numpy as np
from encoders import Encoder




class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, **kwargs):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        # initialise netowrk components
        self.linear = nn.Linear(kwargs.get('nclass', None), nitems, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :param m: mask representing which data_pars is missing
        :return: tensor representing reconstructed item responses
        """
        out = self.linear(x)
        out = self.activation(out)
        return out

class VQDecoder(pl.LightningModule):
    """
    Decoder model for the VQ-VAE (input shape will depend on the size of the latent embedding)
    """

    def __init__(self, nitems: int, **kwargs):
        """
        Initialisation
        :param nitems: the number of items
        :param **kwargs: should contain *emb_dim*
        :param emb_dim: size of the latent embedding
        """
        super().__init__()
        emb_dim = kwargs.get('emb_dim', None)
        # initialise netowrk components
        self.linear = nn.Linear(emb_dim, nitems//2, bias=True)
        self.linear2 = nn.Linear(nitems//2, nitems//2, bias=True)
        self.linear3 = nn.Linear(nitems//2, nitems, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :param m: mask representing which data_pars is missing
        :return: tensor representing reconstructed item responses
        """
        out = F.elu(self.linear(x))
        out = F.elu(self.linear2(out))
        out = self.linear3(out)
        out = self.activation(out)
        return out

SAMPLERS = {'dvae': SpikeAndExp,
            'gs': GumbelSampler,
            'vq': VectorQuantizer,
            'st': StraightThroughSampler,
            'log': LogisticSampler}

DECODERS = {'dvae': Decoder,
            'gs': Decoder,
            'vq': VQDecoder,
            'st': Decoder,
            'log': Decoder}


class LCA(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder.
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 hidden_layer_size: int,
                 learning_rate: float,
                 sampler_type: str,
                 min_temp: float,
                 n_iw_samples: int =1,
                 **kwargs):
        """
        init
        :param dataloader: pytorch dataloader that loads input response patterns
        :param nitems: number of items
        :param hidden_layer_size: number of nodes in the encoder hidden layer
        :param learning_rate: the learning rate
        :param sampler_type: whether to use the VQVAE, DVAE or GSVAE
        :param kwargs:
        """
        super(LCA, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        if sampler_type == 'vq':
            self.latent_dims = kwargs.get('emb_dim', None)
        else:
            self.latent_dims = kwargs.get('nclass')

        self.encoder = Encoder(nitems,
                               self.latent_dims,
                               hidden_layer_size
        )

        self.sampler = SAMPLERS[sampler_type](**kwargs)
        self.Softmax = nn.Softmax(dim=-1)

        self.decoder = DECODERS[sampler_type](nitems, **kwargs)

        self.lr = learning_rate
        self.kl=0
        self.sampler_type = sampler_type
        self.n_samples = n_iw_samples
        self.min_temp = min_temp

    def forward(self, x: torch.Tensor, m: torch.Tensor=None):
        """
        forward pass though the entire network
        :param x: tensor representing response data_pars
        :param m: mask representing which data_pars is missing
        :return: tensor representing a reconstruction of the input response data_pars
        """

        log_pi = self.encoder(x)

        log_pi = log_pi.repeat(self.n_samples, 1,1)


        zeta = self.sampler(log_pi)

        reco = self.decoder(zeta)
        # Calculate the estimated probabilities
        pi = self.Softmax(log_pi)
        return reco, pi, zeta

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data = batch
        X_hat, pi, z = self(data)


        pi = pi.unsqueeze(2)
        z = z.unsqueeze(2)

        loss, _ = self.loss(data, X_hat, pi, z)
        self.log('train_loss', loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def loss(self, input, reco, pi, z):
        # calculate log likelihood

        lll = ((input * reco).clamp(1e-7).log() + ((1 - input) * (1 - reco)).clamp(1e-7).log()).sum(-1, keepdim=True)

        if self.sampler_type == 'vq':
            kl = self.sampler.vq_loss
            loss = (-lll + kl).mean()
            weight = 1
        else:
            kl_type = 'concrete'
            if kl_type == 'categorical':
                # calculate kl divergence
                log_ratio = torch.log(pi * self.latent_dims + 1e-7)
                kl = torch.sum(pi * log_ratio, dim=-1)

                loss = (-lll + kl).mean()
            elif kl_type == 'concrete':
                unif_probs = torch.full_like(pi, 1.0 / pi.shape[-1])

                z = torch.clamp(z, min=1e-8)  # Ensure strictly positive
                z = z / z.sum(-1, keepdim=True)  # Re-normalize for numerical safety

                log_p_theta = dist.RelaxedOneHotCategorical(torch.Tensor([self.sampler.temperature]),
                                                            probs=unif_probs).log_prob(z).sum(-1, keepdim=True)

                log_q_theta_x = dist.RelaxedOneHotCategorical(torch.Tensor([self.sampler.temperature]),
                                                              probs=pi.detach()).log_prob(z).sum(-1, keepdim=True)


                kl = (log_q_theta_x - log_p_theta)  # kl divergence

                # combine into ELBO
                elbo = lll - kl

                with torch.no_grad():
                    weight = (elbo - elbo.logsumexp(dim=0)).exp()
                    if z.requires_grad:
                        z.register_hook(lambda grad: (weight.unsqueeze(-1) * grad).float())



                loss = (-weight * elbo).sum(0).mean()


        return loss, weight

    def on_train_epoch_end(self):
        if self.sampler_type == 'gs':
            self.sampler.temperature = max(self.sampler.temperature * self.sampler.temperature_decay, self.min_temp)

    def fscores(self, batch, n_mc_samples=50):
        data = batch

        if self.n_samples == 1:
            pi = F.softmax(self.encoder(data), -1)
            return pi.unsqueeze(0)
        else:

            scores = torch.empty((n_mc_samples, data.shape[0], self.latent_dims))
            for i in range(n_mc_samples):

                reco, pi, z = self(data)

                loss, weight = self.loss(data, reco, pi.unsqueeze(2), z.unsqueeze(2))
                #z = z[:, :, :, 0]

                idxs = torch.distributions.Categorical(probs=weight.permute(1, 2, 0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]


                # Use gather to select the appropriate elements from z
                output = torch.gather(z.transpose(0, 1), 1,
                                      idxs_expanded).squeeze().detach()  # Shape [10000, latent dims]


                scores[i, :, :] = output

            return scores

    def compute_parameters(self, data):
        """
        compute the log likelihood,
        :param data: data_pars matrix
        :return: the log likelihood of the data_pars, as well as the estimated class- and conditonal probabilities
        """
        data = torch.Tensor(data)

        if self.sampler_type == 'gs':
            pi = self.fscores(torch.Tensor(data)).mean(0)
            est_probs = self.decoder(torch.eye(self.latent_dims)).T
        elif self.sampler_type == 'vq':
            ze = self.encoder(data)
            pred_class_ix = self.sampler.closest_emb_ix(ze)
            pred_class_ix = pred_class_ix
            pi = torch.zeros((data.shape[0], self.sampler.embeddings.num_embeddings), dtype=torch.float)


            pi[torch.arange(data.shape[0]), pred_class_ix] = 1

            embs = self.sampler.embeddings.weight
            est_probs = self.decoder(embs).T 


        log_likelihood = torch.sum(data * torch.log(pi@est_probs.T + 1e-6) +
                                (1 - data) * torch.log(1 - pi@est_probs.T + 1e-6))

        est_probs = est_probs.unsqueeze(1)
        return pi, None, est_probs, log_likelihood


class RestrictedBoltzmannMachine(pl.LightningModule):

    def __init__(self, dataloader, n_visible, n_hidden, learning_rate, n_gibbs):
        super(RestrictedBoltzmannMachine, self).__init__()
        # true
        self.W = torch.nn.Parameter(torch.randn(n_visible, n_hidden)*0.01)
        self.b_hidden = torch.nn.Parameter(torch.zeros(n_hidden))
        self.b_visible = torch.nn.Parameter(torch.zeros(n_visible))

        self.nitems = n_visible

        # hyperparamters
        self.lr = learning_rate
        self.n_gibbs = n_gibbs

        self.dataloader = dataloader
        #self.automatic_optimization = False

    def sample_h(self, v):
        ph = torch.sigmoid(torch.matmul(v, self.W) + self.b_hidden)
        h = torch.bernoulli(ph)

        return h, ph

    def sample_v(self, h):
        pv = torch.sigmoid(torch.matmul(h, self.W.t()) + self.b_visible)
        v = torch.bernoulli(pv)

        return v, pv

    def free_energy(self, v):
        vbias_term = v.mv(self.b_visible)
        h = torch.matmul(v, self.W) + self.b_hidden
        hidden_term = h.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

    def training_step(self, v0):
        v = v0.detach().clone()
        for i in range(self.n_gibbs):
            h, ph = self.sample_h(v)
            v, pv = self.sample_v(h)

        loss = self.free_energy(v0) - self.free_energy(v)

        self.log('train_loss', loss)
        return {'loss': loss}


    def train_dataloader(self):
        return self.dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)


def sim_lca_pars(N, nitems, nclass):
    # sample conditional probabilities
    cond_probs = np.expand_dims(np.random.uniform(.3, .7, nitems), -1).repeat(nclass, -1)
    # Iterate over each row
    for i in range(nclass):
        # Randomly choose the number of entries to set to one
        #num_ones = int(nitems * 0.3)
        # Randomly select num_ones indices to set to one
        #indices = np.random.choice(nitems, num_ones, replace=False)

        # Set the selected indices to one
        cond_probs[:, i] += np.random.uniform(.2 .2, nitems)

    # generate uniform class probabilities #TODO allow for diferent distribtion of class probabilities
    class_probs = np.ones(nclass) / nclass
    # Generate true class membership for each person
    true_class_ix = np.random.choice(np.arange(nclass), size=(N,), p=class_probs)
    true_class = np.zeros((N, nclass))
    true_class[np.arange(N), true_class_ix] = 1

    return cond_probs, true_class

def sim_LCA(N, nitems, nclass, sim_pars=True):

    if sim_pars:
        cond_probs, true_class = sim_lca_pars(N, nitems, nclass)
        cond_probs = np.expand_dims(cond_probs, 1)
        # np.save(f'./saved_data/LCA/itempars/{nclass}_{nitems}.npy', cond_probs)
        # np.save(f'./saved_data/LCA/class/{nclass}_{nitems}.npy', true_class)
        # print(f'parameters saved.')
        # exit()

    else:
        cond_probs = np.load(f'./saved_data/LCA/itempars/{nclass}_{nitems}.npy')
        true_class = np.load(f'./saved_data/LCA/class/{nclass}_{nitems}.npy')


    # simulate responses
    prob = true_class @ cond_probs.squeeze().T
    data = np.random.binomial(1, prob).astype(float)


    return data, true_class, cond_probs
