import pytorch_lightning as pl
import torch.nn.functional as F
import torch.distributions as dist
import torch
from samplers import GumbelSampler
from encoders import *
import numpy as np
from torch.special import digamma, logsumexp

# normalizing constant for dirichlet
def logB_dir(a):
    return torch.lgamma(a).sum() - torch.lgamma(a.sum())

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

        lll = (batch * X_hat.clamp(1e-7, 1 - 1e-7).log() +
               (1 - batch) * (1 - X_hat).clamp(1e-7, 1 - 1e-7).log()).sum(-1, keepdim=True)

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
        data, _ = batch
        X_hat, pi, att = self(data)
        loss, _ = self.loss(X_hat, att, pi, data)

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


class VariationalGDINA(pl.LightningModule):
    """
    Variational EM for GDINA with Q-matrix masking of effects.
      - Latent attribute profiles over all 2^K combinations with Dirichlet prior on class proportions π.
      - Item parameters delta: POINT ESTIMATES (no prior), updated by gradient steps each epoch.
      - Uses your GDINADecoder(Q): columns of Q are effects; we add an intercept column inside the decoder.
    """
    def __init__(self,
                 dataloader,
                 n_items: int,
                 n_attributes: int,
                 Q,                             # (J, E_effects) mask over effects (no intercept column!)
                 alpha0: torch.Tensor = None,   # Dirichlet prior over 2^K profiles (vector or scalar concentration)
                 eps: float = 1e-6,
                 delta_lr: float = 5e-2,
                 delta_steps: int = 20,
                 max_enum_k: int = 20):
        super().__init__()
        self.automatic_optimization = False # optimize manually using conjucagy
        self.dataloader = dataloader
        self.n_items = n_items
        self.n_att = n_attributes
        self.eps = eps

        self.delta_lr = delta_lr
        self.delta_steps = delta_steps


        self.n_effects = 2 ** self.n_att

        self.decoder = GDINADecoder(Q)

        # Dirichlet prior, flat if not specified
        if alpha0 is None:
            alpha0 = torch.ones(self.n_effects, dtype=torch.float32)
        elif alpha0.ndim == 0:
            alpha0 = torch.full((self.n_effects,), float(alpha0))

        self.register_buffer("alpha0", alpha0.float())
        self.register_buffer("alpha", self.alpha0.clone())


        # all_attr: (C, K) binary profiles
        all_eff = torch.tensor(
            [list(map(int, f"{i:0{self.n_att}b}")) for i in range(self.n_effects)],
            dtype=torch.float32
        )
        self.register_buffer("all_attr", all_eff)  # (C, K)

        # effects for every profile
        eff = expand_interactions(all_eff)      # (1, C, S)
        self.register_buffer("all_eff", eff)   # (1, C, S)

        # sufficient stats
        self.register_buffer("Nk_sum", torch.zeros(self.n_effects))
        self.register_buffer("elbo_sum", torch.tensor(0.0))
        self.samples_seen = 0


    def _class_item_probs(self):
        """
        Compute class-conditional item probabilities for all classes at once.
        Returns:
            probs: (C, J)
        """
        # decoder.forward: input (IW, B, S_effects); output (IW, B, J)
        probs = self.decoder(self.all_eff)  # (1, C, J)
        probs = probs.squeeze(0)            # (C, J)
        probs = probs.clamp(self.eps, 1 - self.eps)
        return probs



    def forward(self, X):
        """
        E-step: responsibilities r (N, C) given current alpha and delta.
        Also returns probs (C, J) and per-batch log_like (N, C).
        """
        probs = self._class_item_probs()                      # (C, J)
        log_like = (X @ probs.log().T) + ((1 - X) @ (1 - probs).log().T)  # (N, C)

        Elogpi = digamma(self.alpha) - digamma(self.alpha.sum())          # (C,)
        log_r = log_like + Elogpi
        r = torch.exp(log_r - logsumexp(log_r, dim=1, keepdim=True))      # (N, C)
        return r, probs, log_like, Elogpi


    def training_step(self, batch):
        data, _ = batch
        X = data.float()
        r, probs, log_like, Elogpi = self(X) # E step

        # accumulate sufficient stats for π (we do π-update once per epoch)
        self.Nk_sum += r.sum(0).detach()
        self.samples_seen += X.size(0)

        # ELBO
        batch_elbo = (r * log_like).sum() + (r * Elogpi).sum()
        batch_elbo += -(r.clamp_min(1e-12).log() * r).sum()  # entropy of q(Z)
        self.elbo_sum += batch_elbo.detach()

        # log loss function (negative ELBO / N)
        loss = -batch_elbo / X.size(0)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return {"loss": loss}

    def on_train_epoch_start(self):
        self.Nk_sum.zero_()
        self.elbo_sum.zero_()
        self.samples_seen = 0

    def on_train_epoch_end(self):
        # M step for pi
        self.alpha = self.alpha0 + self.Nk_sum

        # M-step for delta
        opt = torch.optim.Adam([self.decoder.delta], lr=self.delta_lr)
        for _ in range(self.delta_steps): # do so many steps
            for batch in self.train_dataloader():
                Xb = batch[0] if isinstance(batch, (list, tuple)) else (
                batch["x"] if isinstance(batch, dict) else batch)
                Xb = Xb.float().to(self.device)

                # E-step for this batch with current delta & alpha
                p_att, probs, _, _ = self(Xb)    # probs depends on decoder.delta

                # Expected complete-data negative log-likelihood (per batch)
                # Using sufficient stats form:
                rTX = p_att.T @ Xb               # (C, J)
                rT1mX = p_att.T @ (1 - Xb)       # (C, J)
                loss_batch = - (rTX * probs.log() + rT1mX * (1 - probs).log()).sum()

                opt.zero_grad(set_to_none=True)
                loss_batch.backward()
                # enforce mask/normalization via constrain (done implicitly next forward),
                # but keep raw logits unconstrained; no in-place projection needed here
                opt.step()


        Elogpi = digamma(self.alpha) - digamma(self.alpha.sum())
        log_p_pi = -logB_dir(self.alpha0) + ((self.alpha0 - 1) * Elogpi).sum()
        log_q_pi = -logB_dir(self.alpha)  + ((self.alpha  - 1) * Elogpi).sum()
        elbo_epoch = self.elbo_sum + (log_p_pi - log_q_pi)
        self.log("elbo_epoch", elbo_epoch / max(1, self.samples_seen), prog_bar=True)

    def train_dataloader(self):
        return self.dataloader

    def configure_optimizers(self):
        return None # manual optimization

    @torch.no_grad()
    def fscores(self, batch):
        """
        Return per-subject attribute marginal probabilities (N, K),
        obtained by responsibilities over classes times class->attribute map.
        """
        X = batch.float().to(self.device)
        p_att, _, _, _ = self(X)              # (N, C)
        # expectation of attributes u
        pi_attr = p_att @ self.all_attr.to(self.device)  # (N, K)
        return pi_attr

    @torch.no_grad()
    def compute_parameters(self, data):
        """
        Match your amortized `compute_parameters` signature and shapes:
          returns (pi, None, delta, log_likelihood)
          - pi: (N, K) attribute marginals
          - None: placeholder
          - delta: (J, 1, E_effects+1) after constraint (intercept included)
          - log_likelihood: summed scalar log-likelihood
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        X = data.float().to(self.device)

        # E-step for full data
        p_att, probs, _, _ = self(X)                      # r: (N, C), probs: (C, J)

        # attribute marginals (N, K), consistent with your downstream API
        pi = p_att @ self.all_attr.to(self.device)

        # constrained delta with Q mask and softmax over effects (intercept+effects)
        delta = self.decoder.constrain_delta(self.decoder.delta)  # (J, E_with_intercept)
        delta_out = delta.unsqueeze(1)                            # (J, 1, E_with_intercept)

        # mixture probabilities per person and item
        mix_probs = (p_att @ probs).clamp(self.eps, 1 - self.eps)     # (N, J)
        log_likelihood = torch.sum(X * mix_probs.log() + (1 - X) * (1 - mix_probs).log())

        return pi, None, delta_out, log_likelihood
