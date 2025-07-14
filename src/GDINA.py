import pytorch_lightning as pl
import torch.nn.functional as F
import torch.distributions as dist
import torch
from samplers import GumbelSampler
from encoders import *
import numpy as np

class GDINADecoder(pl.LightningModule):
    def __init__(self,
                 Q):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(GDINADecoder, self).__init__()

        # expand the Q matrix so that the columns represent effects instead of attributes
        self.Q = torch.cat((torch.ones(Q.shape[0]).unsqueeze(-1), torch.Tensor(Q)), dim=-1)

        self.Q.requires_grad = False
        self.mask = self.Q != 0


        self.delta = torch.randn(self.Q.shape[0], self.Q.shape[1], requires_grad=True)


    def constrain_delta(self, delta):

        masked_delta = delta.masked_fill(~self.Q.bool(), float('-inf'))

        delta = F.softmax(masked_delta, dim=-1)
        # Replace masked elements with zero in the final result
        delta = delta * self.Q

        return delta


    def forward(self, Z) -> torch.Tensor:

        Z = torch.cat((torch.ones(Z.shape[1]).repeat(Z.shape[0],1).unsqueeze(-1), Z), dim=-1)

        delta = self.constrain_delta(self.delta)


        probs = Z @ delta.T

        return probs


class GDINA(pl.LightningModule):
    def __init__(self, n_items, n_attributes, dataloader, Q, learning_rate, min_temp, n_iw_samples, **kwargs):
        super(GDINA, self).__init__()
        self.n_attributes = n_attributes
        self.n_items = n_items

        self.encoder = Encoder(n_items, self.n_attributes*2, self.n_attributes*2)
        #self.encoder = GDINAEncoder(n_items, self.n_attributes)
        self.sampler = GumbelSampler(**kwargs)

        self.decoder = GDINADecoder(Q)

        self.dataloader = dataloader
        self.lr = learning_rate
        self.min_temp = min_temp
        self.n_samples = n_iw_samples


    def forward(self, X):
        logits = self.encoder(X)

        logits = torch.nn.functional.softplus(logits.reshape((logits.shape[0], logits.shape[1] // 2, 2)))
        #logits = (logits - logits.max(dim=-1, keepdim=True).values).exp()  # Subtract max for stability


        logits = logits.repeat(self.n_samples, 1, 1,1)

        #probs = F.softmax(logits, dim=-1)

        att = self.sampler(logits)
        att = att / att.sum(-1, keepdim=True) # make sure probabilities sum to one (sometimes not true due to numerical issues)
        att = att.clamp(1e-5, 1-1e-5)

        eff = expand_interactions(att[:, :,:, 0])

        #x_hat = self.decoder(att[:, :, :, 0])

        x_hat = self.decoder(eff)
        pi = F.softmax(logits, dim=-1)

        return x_hat, pi, att

    def loss(self, X_hat, z, pi, batch):

        lll = ((batch * X_hat).clamp(1e-7).log() + ((1 - batch) * (1 - X_hat)).clamp(1e-7).log()).sum(-1, keepdim=True)

        # Compute the KL divergence for each attribute
        kl_type = 'concrete'
        if kl_type=='categorical':

            bce = torch.nn.functional.binary_cross_entropy(X_hat.squeeze(), batch, reduction='none')
            bce = torch.mean(bce) * self.n_items
            # Analytical KL based on categorical distribution
            kl_div = pi * torch.log(pi / 0.5) + (1 - pi) * torch.log((1 - pi) / 0.5)

            # Sum KL divergence over the latent variables (dimension K) and average over the batch
            kl_div = torch.sum(kl_div, dim=1)  # Sum over K latent variables
            kl_div = torch.mean(kl_div)  # Mean over the batch

            loss = (bce + kl_div)
            weight = 1
        elif kl_type=='concrete':
            # prior probability of samples p(z)

            #print(f'min {torch.min(z)}')
            #print(f'max {torch.max(z)}')
            z = torch.clamp(z, min=1e-8)  # Ensure strictly positive
            z = z / z.sum(-1, keepdim=True)  # Re-normalize for numerical safety
            log_p_theta = dist.RelaxedOneHotCategorical(torch.Tensor([self.sampler.temperature]),
                                                        probs=torch.ones_like(pi)).log_prob(z).sum(-1)
            # posterior probability of the samples p(z|x)
            log_q_theta_x = dist.RelaxedOneHotCategorical(torch.Tensor([self.sampler.temperature]),
                                                          probs=pi.detach()).log_prob(z).sum(-1)

            # kl divergence
            kl = (log_q_theta_x - log_p_theta).unsqueeze(-1)  # kl divergence

            # compute elbo
            elbo = lll - kl

            # do importance weighting
            with torch.no_grad():
                weight = (elbo - elbo.logsumexp(dim=0)).exp()
                if z.requires_grad:
                    z.register_hook(lambda grad: (weight.unsqueeze(-1) * grad).float())
            loss = (-weight * elbo).sum(0).mean()

        return loss, weight

    def training_step(self, batch):
        X_hat, pi, att = self(batch)
        loss, _ = self.loss(X_hat, att, pi, batch)

        self.log('train_loss', loss)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam([
                {'params': self.encoder.parameters(), 'lr': self.lr},
                {'params': self.decoder.delta, 'lr': self.lr}
            ],
            amsgrad=True
        )

        return [optimizer]

    def on_train_epoch_end(self):
        self.sampler.temperature = max(self.sampler.temperature * self.sampler.temperature_decay, self.min_temp)

    def fscores(self, batch, n_mc_samples=50):
        data = batch

        if self.n_samples == 1:
            logits = self.encoder(data)
            logits = torch.nn.functional.softplus(logits.reshape((logits.shape[0], logits.shape[1] // 2, 2)))
            #logits = logits.reshape((logits.shape[0], logits.shape[1] // 2, 2)).exp()
            mu = self.sampler.softmax(logits)[:, :, 0]
            return mu.unsqueeze(0)
        else:

            scores = torch.empty((n_mc_samples, data.shape[0], self.n_attributes))
            for i in range(n_mc_samples):

                reco, pi, z = self(data)

                loss, weight = self.loss(reco, z, pi, data)
                z = z[:, :, :, 0]

                idxs = torch.distributions.Categorical(probs=weight.permute(1, 2, 0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]


                # Use gather to select the appropriate elements from z
                output = torch.gather(z.transpose(0, 1), 1,
                                      idxs_expanded).squeeze().detach()  # Shape [10000, latent dims]
                if self.n_attributes == 1:
                    output = output.unsqueeze(-1)

                scores[i, :, :] = output

            return scores

    def compute_parameters(self, data):
        data = torch.Tensor(data)
        pi = self.fscores(torch.Tensor(data)).mean(0)


        pi_att = expand_interactions(pi).squeeze()
        pi_att = torch.cat((torch.ones(pi_att.shape[0]).unsqueeze(-1), pi_att), -1)

        delta = self.decoder.delta
        delta = self.decoder.constrain_delta(delta)


        log_likelihood = torch.sum(data * torch.log(pi_att @ delta.T + 1e-6) +
                                   (1 - data) * torch.log(1 - (pi_att @ delta.T - 1e-6)))


        delta = delta.unsqueeze(1)


        return pi, None,  delta, log_likelihood

def expand_interactions(attributes):
    """
    Function that computes all possible latent classes given the different combinations of attributes
    :param: attributes: Tensor of dim IW samples x batch_size x N-attributes
    :returns: Tensor of IW samples x batch_size x 2** N-attributes-1, representing all possible combinations
    """
    # make sure the attributes have 3 dimensions (IW samples, batch size, n_attributes)
    if len(attributes.shape) == 2:
        attributes = attributes.unsqueeze(0)

    n_iw_samples = attributes.shape[0]
    n_attributes = attributes.shape[2]
    total_effects = 2**n_attributes-1
    n_effects = n_attributes + n_attributes*(n_attributes-1)//2
    batch_size = attributes.shape[1]


    # Generate SxA matrix where each row represents whether each attribute is needed for each effect
    required_mask = torch.arange(1, total_effects + 1).unsqueeze(1).bitwise_and(1 << torch.arange(n_attributes)).bool()

    required_mask = required_mask[required_mask.sum(-1) <= 2,] # only keep 1st and second order



    # repeat the matrix for each IW sample and each observation
    required_mask = required_mask.repeat((n_iw_samples, batch_size, 1, 1))  # IWxBxSxA

    # repeat the observed attribute pattern for each possible combination
    attributes = attributes.unsqueeze(2).repeat(1, 1, n_effects, 1)

    # set the observed attributes to 1 if they are not required for a pattern
    attributes[~required_mask] = 1

    # multiply over the diffent attributes, so that we get the probability of observing all necessary attributes
    effects = attributes.prod(3)




    return effects

def sim_gdina_pars(N, nitems, nattributes):
    neffects = nattributes + (nattributes*(nattributes-1)) // 2
    Q_start = torch.eye(nattributes)  # make sure each attribute has at least one unique item
    Q_rest = torch.zeros((nitems - nattributes, nattributes))
    # Q = torch.zeros((nitems, nattributes))

    valid_Q = False
    while not valid_Q:
        for item in range(nitems - nattributes):
            n_attr_it = torch.randint(1, min(nattributes, 3) + 1, (1,)).item()
            atts = torch.randperm(nattributes)[:(n_attr_it)]
            Q_rest[item, atts] = 1
            Q = torch.cat((Q_start, Q_rest), dim=0)
        if torch.all(Q.sum(dim=0) >= 3):
            valid_Q = True

    att = torch.bernoulli(torch.full((N, nattributes), 0.5))


    delta = torch.rand(nitems, neffects)
    delta *= expand_interactions(torch.Tensor(Q)).squeeze()
    #intercepts = torch.rand(nitems) * 0.2
    delta /= (delta.sum(axis=1, keepdims=True) + 1e-4)

    intercepts = np.zeros(nitems)

    delta = np.column_stack((intercepts, delta))

    return att, delta


def sim_GDINA(N, nitems, nattributes, sim_pars):
    if sim_pars:
        att, delta = sim_gdina_pars(N, nitems, nattributes)
        # np.save(f'./saved_data/LCA/class/{nattributes}_{nitems}.npy', att)
        # np.save(f'./saved_data/LCA/itempars/{nattributes}_{nitems}.npy', delta)
    else:
        att = torch.Tensor(np.load(f'../saved_data/GDINA/class/{nattributes}_{nitems}.npy'))
        delta = torch.Tensor(np.load(f'../saved_data/GDINA/itempars/{nattributes}_{nitems}.npy')).squeeze()

    eff = expand_interactions(att).squeeze()
    eff = torch.column_stack((torch.ones(N), eff))

    probs = eff @ delta.T

    data = np.random.binomial(1, probs).astype(float)
    delta = np.expand_dims(delta, 1)

    return data, delta, att.detach().numpy()