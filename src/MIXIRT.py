import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
from samplers import NormalSampler,  GumbelSampler
from encoders import Encoder
import numpy as np
from torch.special import digamma, logsumexp

class IRTDecoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int,  qm: torch.Tensor=None, nclass=2):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()
        # one layer for each class
        self.weights = nn.Parameter(torch.zeros((latent_dims, nitems)))  # Manually created weight matrix

        self.biases = nn.Parameter(torch.zeros(nclass, nitems))  # shape: [2, nitems]
        self.activation = nn.Sigmoid()
        self.nclass = nclass

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
        self.qm = self.qm.to(self.weights)
        pruned_weights= self.weights * self.qm

        out = torch.matmul(theta, pruned_weights) + torch.matmul(cl, self.biases)
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
        assert (nclass == 2 or nclass ==1), 'mixture has only been tested for one or two classes'
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

        self.decoder = IRTDecoder(nitems, latent_dims, qm, nclass)

        self.nclass = nclass

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
        cl = latent_vector[:,(self.latent_dims*2):(self.latent_dims*2+self.nclass)]


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
        #cl = cl / cl.sum(dim=-1, keepdim=True)

        cl = torch.clamp(cl, min=1e-6, max=1-1e-6)  # Ensure strictly positive
        cl = cl / cl.sum(-1, keepdim=True)  # Re-normalize for numerical safety


        # calculate normal KL divergence
        log_q_theta_x = torch.distributions.Normal(mu.detach(), sigma.exp().detach()+ 1e-7).log_prob(z).sum(dim = -1, keepdim = True) # log q(Theta|X)
        log_p_theta = torch.distributions.Normal(torch.zeros_like(z).to(input), scale=torch.ones(mu.shape[2]).to(input)).log_prob(z).sum(dim = -1, keepdim = True) # log p(Theta)
        kl_normal =  log_q_theta_x - log_p_theta # kl divergence

        # calculate concrete KL divergence
        #pi = torch.clamp(pi, min=1e-7, max=1 - 1e-7)
        #cl = torch.clamp(cl, min=1e-7, max=1 - 1e-7)

        unif_probs = torch.full_like(pi, 1.0 / pi.shape[-1])
        log_p_cl = torch.distributions.RelaxedOneHotCategorical(torch.Tensor([self.GumbelSoftmax.temperature]).to(pi),
                                                    probs=unif_probs).log_prob(cl).unsqueeze(-1)

        log_q_cl_x = torch.distributions.RelaxedOneHotCategorical(torch.Tensor([self.GumbelSoftmax.temperature]).to(pi),
                                                      probs=pi.detach()).log_prob(cl).unsqueeze(-1)

        kl_concrete = (log_q_cl_x - log_p_cl)

        # combine into ELBO
        elbo = logll - kl_normal  - kl_concrete


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

    def fscores(self, batch, n_mc_samples=50):
        data = batch

        if self.n_samples == 1:
            latent_vector = self.encoder(data)
            mu = latent_vector[:, 0:self.latent_dims]
            log_sigma = latent_vector[:, self.latent_dims:(self.latent_dims * 2)]
            cl = latent_vector[:, (self.latent_dims * 2):(self.latent_dims * 2 + 2)]
            return mu.unsqueeze(0).detach(), cl.unsqueeze(0).detach()
        else:
            scores = torch.empty((n_mc_samples, data.shape[0], self.latent_dims))
            classes = torch.empty((n_mc_samples, data.shape[0], self.nclass))
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
                if cl_output.dim() ==1:
                    cl_output = cl_output.unsqueeze(-1)
                classes[i, :, :] = cl_output

            return scores, classes

    def compute_parameters(self, data):
        data = torch.Tensor(data)
        a_est = self.decoder.weights.detach()
        b_est = self.decoder.biases.detach()


        theta_est, cl_est = self.fscores(data)
        theta_est = theta_est.mean(0)
        cl_est = cl_est.mean(0)


        logits = torch.matmul(theta_est, a_est) + torch.matmul(cl_est, b_est)
        # logits = cl_est[:,[0]] * (theta_est @ a_est + d1_est) + cl_est[:,[1]] * (theta_est @ a_est + d2_est)
        probs = F.sigmoid(logits)
        epsilon = 1e-6  # Small constant to avoid log(0)
        probs = torch.clamp(probs, epsilon, 1 - epsilon)

        log_likelihood = torch.sum(data * probs.log() + (1-data) * (1-probs).log())

        a_rep = a_est.T.unsqueeze(-1).expand(-1, -1, self.nclass)  # shape (nitems, ndim, nclass)
        b_exp = b_est.T.unsqueeze(1)  # shape (nitems, 1, nclass)
        itempars = torch.cat([b_exp, a_rep], dim=1) # shape (nitems, ndim+1, nclass)

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
    true_difficulty[:,1] += np.random.normal(0,1, true_difficulty.shape[0])
    true_difficulty[:, 0] += np.random.normal(0, 1, true_difficulty.shape[0])
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




class VariationalMixMIRT(pl.LightningModule):
    """
    Neural network for standard (non-amortized) variational inference:
    - per-observation variational params (mu, log_sigma, class logits) are learned directly
    - global item parameters are learned here (formerly in IRTDecoder)
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 n_iw_samples: int,
                 temperature: float,
                 temperature_decay: float,
                 min_temp: float,
                 n_obs: int,
                 nclass: int = 2,
                 beta: int = 1):
        super(VariationalMixMIRT, self).__init__()
        assert (nclass == 2 or nclass == 1), 'mixture has only been tested for one or two classes'
        self.save_hyperparameters(ignore=['dataloader', 'qm'])  # optional convenience
        self.nitems = nitems
        self.dataloader = dataloader

        self.n_obs = n_obs

        # variational parameters for each observation
        self.mu_param         = nn.Parameter(torch.zeros(n_obs, latent_dims))
        self.log_sigma_param  = nn.Parameter(torch.zeros(n_obs, latent_dims))
        self.class_logits_param = nn.Parameter(torch.zeros(n_obs, nclass))

        # item parameters
        self.weights = nn.Parameter(torch.zeros((latent_dims, nitems)))    # slopes
        self.biases  = nn.Parameter(torch.zeros(nclass, nitems))           # intercepts (calss specific)
        self.activation = nn.Sigmoid()
        self.nclass = nclass

        if qm is None:
            self.qm = torch.ones((latent_dims, nitems))
        else:
            self.qm = torch.as_tensor(qm).t()

        # Gumbel-Softmax temperature schedule (kept)
        self.temperature = float(temperature)
        self.temperature_decay = float(temperature_decay)
        self.min_temp = float(min_temp)

        self.latent_dims = latent_dims
        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.n_samples = n_iw_samples
        self.kl = 0

    def _gumbel_softmax(self, logits, temperature):
        gumbel_noise = -torch.empty_like(logits).exponential_().log()  # sample Gumbel(0,1)
        y = (logits + gumbel_noise) / max(temperature, 1e-8)
        return F.softmax(y, dim=-1)


    def forward(self, x: torch.Tensor, idx: torch.Tensor = None):
        """
        Forward pass through the model using local (per-observation) variational parameters.
        x:  [B, I]
        idx: [B] indices of rows in the dataset corresponding to x.
        """
        if idx is None:
            idx = torch.arange(x.shape[0], device=x.device)

        #  variational params
        mu        = self.mu_param[idx]             # [B, D]
        log_sigma = self.log_sigma_param[idx]      # [B, D]
        log_pi    = self.class_logits_param[idx]   # [B, C]

        # Repeat for importance samples
        mu        = mu.unsqueeze(0).expand(self.n_samples, -1, -1)
        log_sigma = log_sigma.unsqueeze(0).expand(self.n_samples, -1, -1)
        log_pi    = log_pi.unsqueeze(0).expand(self.n_samples, -1, -1)

        # Sample gumbel class assignments
        cl = self._gumbel_softmax(log_pi, self.temperature)  # [K, B, C]

        # Reparameterize theta
        eps = torch.randn_like(mu)
        z = mu + eps * (log_sigma.exp())

        # Likelihood parameters (apply Q-mask like the old decoder)
        qm = self.qm.to(self.weights)
        pruned_weights = self.weights * qm  # [D, I]


        logits = torch.matmul(z, pruned_weights) + torch.matmul(cl, self.biases)
        reco = self.activation(logits)

        pi = F.softmax(log_pi, dim=-1)  # [K, B, C]
        return reco, mu, log_sigma, z, pi, cl

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # Allow either batch or (batch, idx) with minimal changes
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, idx = batch
        else:
            data, idx = batch, None

        reco, mu, log_sigma, z, pi, cl = self(data, idx)
        mask = torch.ones_like(data)
        loss, _ = self.loss(data, reco, mask, mu, log_sigma, z, pi, cl)
        self.log('train_loss', loss)
        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def loss(self, input, reco, mask, mu, sigma, z, pi, cl):
        # same as original, with tiny numeric safety tweaks
        input = input.unsqueeze(0).repeat(reco.shape[0], 1, 1)  # [K, B, I]

        # Bernoulli log-likelihood
        eps = 1e-7
        log_p_x_theta = (input * (reco.clamp(eps, 1 - eps)).log() +
                         (1 - input) * ((1 - reco).clamp(eps, 1 - eps)).log())
        logll = (log_p_x_theta * mask).sum(dim=-1, keepdim=True)  # [K, B, 1]

        # Normalize cl for safety
        cl = torch.clamp(cl, min=1e-6, max=1 - 1e-6)
        cl = cl / cl.sum(-1, keepdim=True)

        # KL standard Normal prior
        log_q_theta_x = torch.distributions.Normal(mu.detach(), sigma.exp().detach() + 1e-7).log_prob(z).sum(dim=-1, keepdim=True)
        log_p_theta   = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1, keepdim=True)
        kl_normal = (log_q_theta_x - log_p_theta)  # [K, B, 1]

        # KL for relaxed one-hot categorical against uniform prior
        unif_probs = torch.full_like(pi, 1.0 / pi.shape[-1])
        temp = torch.tensor([self.temperature], device=pi.device, dtype=pi.dtype)
        log_p_cl = torch.distributions.RelaxedOneHotCategorical(temp, probs=unif_probs).log_prob(cl).unsqueeze(-1)
        log_q_cl_x = torch.distributions.RelaxedOneHotCategorical(temp, probs=pi.detach()).log_prob(cl).unsqueeze(-1)
        kl_concrete = (log_q_cl_x - log_p_cl)  # [K, B, 1]

        # ELBO
        elbo = logll - kl_normal - kl_concrete

        # IW weighting
        with torch.no_grad():
            weight = (elbo - elbo.logsumexp(dim=0)).exp()
            if cl.requires_grad:
                cl.register_hook(lambda grad: (weight * grad).float())
            if z.requires_grad:
                z.register_hook(lambda grad: (weight * grad).float())

        loss = (-weight * elbo).sum(0).mean()
        return loss, weight

    def on_train_epoch_end(self):
        self.temperature = max(self.temperature * self.temperature_decay, self.min_temp)

    def fscores(self, batch, n_mc_samples=50):
        # Return posterior samples/means for z and class probs for a given batch (or (batch, idx))
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, idx = batch
        else:
            data, idx = batch, None

        if self.n_samples == 1:
            # single-sample: just return current variational params
            if idx is None:
                idx = torch.arange(data.shape[0], device=data.device)
            mu = self.mu_param[idx]
            cl_logits = self.class_logits_param[idx]
            return mu.unsqueeze(0).detach(), cl_logits.unsqueeze(0).detach()
        else:
            if idx is None:
                idx = torch.arange(data.shape[0], device=data.device)

            scores = torch.empty((n_mc_samples, data.shape[0], self.latent_dims), device=data.device)
            classes = torch.empty((n_mc_samples, data.shape[0], self.nclass), device=data.device)
            for i in range(n_mc_samples):
                reco, mu, log_sigma, z, pi, cl = self(data, idx)

                mask = torch.ones_like(data)
                loss, weight = self.loss(data, reco, mask, mu, log_sigma, z, pi, cl)

                idxs = torch.distributions.Categorical(probs=weight.permute(1, 2, 0)).sample().long()
                idxs_expanded_z = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))
                idxs_expanded_cl = idxs.unsqueeze(-1).expand(-1, -1, cl.size(2))

                z_output = torch.gather(z.transpose(0, 1), 1, idxs_expanded_z).squeeze().detach()
                cl_output = torch.gather(cl.transpose(0, 1), 1, idxs_expanded_cl).squeeze().detach()

                if self.latent_dims == 1:
                    z_output = z_output.unsqueeze(-1)
                if cl_output.dim() == 1:
                    cl_output = cl_output.unsqueeze(-1)

                scores[i, :, :] = z_output
                classes[i, :, :] = cl_output
            return scores, classes

    def compute_parameters(self, data):
        # Use current global item parameters and average posterior stats
        data = torch.as_tensor(data, dtype=self.mu_param.dtype, device=self.mu_param.device)
        a_est = self.weights.detach()
        b_est = self.biases.detach()

        theta_est, cl_est = self.fscores(data)
        theta_est = theta_est.mean(0)
        cl_est = cl_est.mean(0)

        logits = torch.matmul(theta_est, a_est) + torch.matmul(cl_est, b_est)
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

        log_likelihood = torch.sum(data * probs.log() + (1 - data) * (1 - probs).log())

        a_rep = a_est.T.unsqueeze(-1).expand(-1, -1, self.nclass)  # (nitems, ndim, nclass)
        b_exp = b_est.T.unsqueeze(1)                                # (nitems, 1, nclass)
        itempars = torch.cat([b_exp, a_rep], dim=1)                 # (nitems, ndim+1, nclass)

        return cl_est, theta_est, itempars, log_likelihood