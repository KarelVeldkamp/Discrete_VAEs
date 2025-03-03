import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
from samplers import NormalSampler,  GumbelSampler
from encoders import Encoder
import numpy as np


class IRTDecoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int,  qm: torch.Tensor=None):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()
        # one layer for each class
        self.weights1 = nn.Parameter(torch.zeros((latent_dims, nitems)))  # Manually created weight matrix
        self.bias1 = nn.Parameter(torch.zeros(nitems))  # Manually created bias vecto
        self.weights2 = nn.Parameter(torch.zeros((latent_dims, nitems)))  # Manually created weight matrix
        self.bias2 = nn.Parameter(torch.zeros(nitems))  # Manually created bias vecto
        self.activation = nn.Sigmoid()

        # remove edges between latent dimensions and items that have a zero in the Q-matrix
        if qm is None:
            self.qm = torch.ones((latent_dims, nitems))
        else:
            self.qm = torch.Tensor(qm).t()


    def forward(self, cl: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :param m: mask representing which data_pars is missing
        :return: tensor representing reconstructed item responses
        """

        #print(cl[:, 0:1])
        self.qm = self.qm.to(self.weights1)
        pruned_weights1 = self.weights1 * self.qm
        pruned_weights2 = self.weights2 * self.qm

        #indices = torch.Tensor([1,2,3,4,5,11,12,13,14,15, 21,22, 23,24,25]).int()
        bias1 = self.bias1
        bias2 = self.bias2#1.clone()  # Start with bias1 as the base
        #bias2[indices] = self.bias2[indices]

        out = (torch.matmul(theta, pruned_weights1) + bias1) * cl[:, :, 0:1] + \
            (torch.matmul(theta, pruned_weights1) + bias2) * cl[:, :, 1:2]
        out = self.activation(out)
        return out




class VAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 n_iw_samples: int,
                 temperature:float,
                 temperature_decay:float,
                 min_temp:float,
                 nclass: int = 2,
                 beta: int = 1):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VAE, self).__init__()
        assert nclass == 2, 'mixture only implemented for two classes'
        #self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               latent_dims+latent_dims+nclass,
                               hidden_layer_size
        )

        self.GumbelSoftmax = GumbelSampler(temperature=temperature,
                                           temperature_decay=temperature_decay)

        self.min_temp = min_temp
        self.sampler = NormalSampler()
        self.latent_dims = latent_dims

        self.decoder = IRTDecoder(nitems, latent_dims, qm)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.n_samples = n_iw_samples
        self.kl=0

    def forward(self, x: torch.Tensor, m: torch.Tensor=None):
        """
        forward pass though the entire network
        :param x: tensor representing response data_pars
        :param m: mask representing which data_pars is missing
        :return: tensor representing a reconstruction of the input response data_pars
        """
        latent_vector = self.encoder(x)
        mu = latent_vector[:,0:self.latent_dims]
        log_sigma = latent_vector[:,self.latent_dims:(self.latent_dims*2)]
        cl = latent_vector[:,(self.latent_dims*2):(self.latent_dims*2+2)]


        mu = mu.repeat(self.n_samples,1,1)
        log_sigma = log_sigma.repeat(self.n_samples,1,1)
        log_pi = cl.repeat(self.n_samples,1,1)

        # print(torch.min(log_pi))
        # print(torch.max(log_pi))
        # print(torch.any(torch.isnan(log_pi)))
        cl = self.GumbelSoftmax(log_pi)

        z = self.sampler(mu, log_sigma)

        reco = self.decoder(cl, z)

        pi = F.softmax(log_pi, dim=-1)

        return reco, mu, log_sigma, z, pi, cl

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass
        data = batch
        reco, mu, log_sigma, z, pi, cl = self(data)

        mask = torch.ones_like(data)
        loss, _ = self.loss(data, reco, mask, mu, log_sigma, z, pi, cl)
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def loss(self, input, reco, mask, mu, sigma, z, pi, cl):
        #calculate log likelihood

        input = input.unsqueeze(0).repeat(reco.shape[0], 1, 1) # repeat input k times (to match reco size)
        log_p_x_theta = ((input * reco).clamp(1e-7).log() + ((1 - input) * (1 - reco)).clamp(1e-7).log()) # compute log ll
        logll = (log_p_x_theta * mask).sum(dim=-1, keepdim=True) # set elements based on missing data_pars to zero
        #
        cl = cl / cl.sum(dim=-1, keepdim=True)
        # calculate normal KL divergence
        log_q_theta_x = torch.distributions.Normal(mu.detach(), sigma.exp().detach()+ 1e-7).log_prob(z).sum(dim = -1, keepdim = True) # log q(Theta|X)
        log_p_theta = torch.distributions.Normal(torch.zeros_like(z).to(input), scale=torch.ones(mu.shape[2]).to(input)).log_prob(z).sum(dim = -1, keepdim = True) # log p(Theta)
        kl_normal =  log_q_theta_x - log_p_theta # kl divergence

        # calculate concrete KL divergence
        pi = torch.clamp(pi, min=1e-7, max=1 - 1e-7)
        cl = torch.clamp(cl, min=1e-7, max=1 - 1e-7)



        log_p_cl = torch.distributions.RelaxedOneHotCategorical(torch.Tensor([self.GumbelSoftmax.temperature]).to(pi),
                                                    probs=torch.ones_like(pi)).log_prob(cl).unsqueeze(-1)

        log_q_cl_x = torch.distributions.RelaxedOneHotCategorical(torch.Tensor([self.GumbelSoftmax.temperature]).to(pi),
                                                      probs=pi).log_prob(cl).unsqueeze(-1)

        kl_concrete = (log_q_cl_x - log_p_cl)



        # combine into ELBO
        elbo = logll - kl_normal - kl_concrete


        # # perform importance weighting
        with torch.no_grad():
            weight = (elbo - elbo.logsumexp(dim=0)).exp()

            if cl.requires_grad:
                cl.register_hook(lambda grad: (weight * grad).float())
            if z.requires_grad:
                z.register_hook(lambda grad: (weight * grad).float())
        #
        loss = (-weight * elbo).sum(0).mean()

        return loss, weight

    def on_train_epoch_end(self):
        self.GumbelSoftmax.temperature = max(self.GumbelSoftmax.temperature * self.GumbelSoftmax.temperature_decay, self.min_temp)

    def fscores(self, batch, n_mc_samples=500):
        data = batch

        if self.n_samples == 1:
            mu, _, cl = self.encoder(data)
            return mu.unsqueeze(0), cl.unsqueeze(0)
        else:
            scores = torch.empty((n_mc_samples, data.shape[0], self.latent_dims))
            classes = torch.empty((n_mc_samples, data.shape[0], 2))
            for i in range(n_mc_samples):
                reco, mu, log_sigma, z, pi, cl = self(data)
                mask = torch.ones_like(data)
                loss, weight = self.loss(data, reco, mask, mu, log_sigma, z, pi, cl)

                idxs = torch.distributions.Categorical(probs=weight.permute(1,2,0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded_z = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]
                idxs_expanded_cl = idxs.unsqueeze(-1).expand(-1, -1, cl.size(2))  # Shape [10000, 1, 2]

                # Use gather to select the appropriate elements from z
                z_output = torch.gather(z.transpose(0, 1), 1, idxs_expanded_z).squeeze().detach() # Shape [10000, latent dims]

                cl_output = torch.gather(cl.transpose(0, 1), 1,
                                      idxs_expanded_cl).squeeze().detach()  # Shape [10000, latent dims]
                if self.latent_dims == 1:
                    cl_output = cl_output.unsqueeze(-1)
                    z_output =  z_output.unsqueeze(-1)

                scores[i, :, :] = z_output
                classes[i, :, :] = cl_output

            return scores, classes

    def compute_parameters(self, data):
        data = torch.Tensor(data)
        a1_est = self.decoder.weights1.detach()
        a2_est = self.decoder.weights1.detach()
        d1_est = self.decoder.bias1.detach()
        d2_est = self.decoder.bias2.detach()

        theta_est, cl_est = self.fscores(data)
        theta_est = theta_est.mean(0)
        cl_est = cl_est.mean(0)
        #theta_est = latent_samples[:, 0:self.latent_dims]
        #cl = latent_samples[:, (2*self.latent_dims)+2]

        logits = cl_est[:,[0]] * (theta_est @ a1_est + d1_est) + cl_est[:,[1]] * (theta_est @ a2_est + d2_est)
        probs = F.sigmoid(logits)

        log_likelihood = torch.sum(data * probs.log() + (1-data) * (1-probs).log())


        items1 = torch.cat((d1_est.unsqueeze(-1),a1_est.T), -1)
        items2 = torch.cat((d2_est.unsqueeze(-1),a2_est.T), -1)


        itempars = torch.cat((items1.unsqueeze(-1), items2.unsqueeze(-1)), -1)


        return cl_est, theta_est, itempars, log_likelihood

def sim_mixirt_pars(N, nitems, nclass, mirt_dim, Q, class_prob=.5, cov=0):
    # Step 1: Creating true_class tensor with torch
    true_class_ix = np.random.binomial(1, class_prob, N)
    # Convert to one-hot encoding
    true_class = np.eye(2)[true_class_ix]


    covMat = np.full((mirt_dim, mirt_dim), cov)  # covariance matrix of dimensions, zero for now
    np.fill_diagonal(covMat, 1)
    true_theta = np.random.multivariate_normal([0] * mirt_dim, covMat, N)
    true_difficulty = np.repeat(np.random.uniform(-2, 2, (nitems, 1)), 2, axis=1)
    true_difficulty[:,1] += np.random.normal(0,.2, true_difficulty.shape[0])
    #true_difficulty = np.random.uniform(-2, 2, (nitems, 2))


    # true_slopes = np.random.uniform(.5, 2, (cfg['nitems'], cfg['mirt_dim'],2))
    true_slopes = np.repeat(np.random.uniform(.5, 2, (nitems, mirt_dim, 1)), 2, -1)
    true_slopes *= np.expand_dims(Q, -1)


    true_itempars = np.concatenate((true_difficulty[:, np.newaxis, :], true_slopes), axis=1)

    return true_theta, true_class, true_itempars


def sim_MIXIRT(N, nitems, nclass, mirt_dim, Q, class_prob=.5, cov=0, sim_pars=False):
    if sim_pars:
        true_theta, true_class, true_itempars = sim_mixirt_pars(N, nitems, nclass, mirt_dim, Q, class_prob, cov)
        # np.save(f'./saved_data/MIXIRT/theta/{mirt_dim}.npy', true_theta)
        # np.save(f'./saved_data/MIXIRT/class/{mirt_dim}.npy', true_class)
        # np.save(f'./saved_data/MIXIRT/itempars/{mirt_dim}.npy', true_itempars)

    else:
        true_theta = np.load(f'./saved_data/MIXIRT/theta/{mirt_dim}.npy')
        true_class = np.load(f'./saved_data/MIXIRT/class/{mirt_dim}.npy')
        true_itempars = np.load(f'./saved_data/MIXIRT/itempars/{mirt_dim}.npy')


    exponent = ((np.dot(true_theta, true_itempars[:, 1:(mirt_dim+1), 0].T) + true_itempars[:, 0, 0]) * (true_class[:,[0]]) + \
                (np.dot(true_theta, true_itempars[:, 1:(mirt_dim+1), 1].T) + true_itempars[:, 0, 1]) * (true_class[:,[1]]))

    prob = np.exp(exponent) / (1 + np.exp(exponent))
    data = np.random.binomial(1, prob).astype(float)
    true_class = np.squeeze(true_class)

    return data, true_class, true_theta, true_itempars