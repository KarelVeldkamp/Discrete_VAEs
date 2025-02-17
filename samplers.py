import pytorch_lightning as pl
import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F

class GumbelSampler(pl.LightningModule):
    """
    Network module that performs the Gumbel-Softmax operation
    """
    def __init__(self,  **kwargs):
        """
        initialize
        :param temperature: temperature parameter at the start of training
        :param temperature_decay: factor to multiply temperature by each epoch
        """
        super(GumbelSampler, self).__init__()
        # Gumbel distribution
        self.softmax = torch.nn.Softmax(dim=2)
        self.temperature = kwargs.get('temperature', None)
        self.temperature_decay = kwargs.get('temperature_decay', None)

    def forward(self, logits):
            """
            forward pass
            :param log_pi: NxM tensor of log probabilities where N is the batch size and M is the number of classes
            :return: NxM tensor of 'discretized probabilities' the lowe the temperature the more discrete
            """

            # Define the temperature for the RelaxedOneHotCategorical
            temperature = self.temperature

            # Create the RelaxedOneHotCategorical distribution
            distribution = dist.RelaxedOneHotCategorical(temperature, logits=logits)

            # Sample from the distribution
            samples = distribution.rsample()

            return samples



class LogisticSampler(pl.LightningModule):
    """
    Network module that performs the Gumbel-Softmax operation
    """
    def __init__(self, **kwargs):
        """
        initialize
        :param temperature: temperature parameter at the start of training
        :param temperature_decay: factor to multiply temperature by each epoch
        """
        super(LogisticSampler, self).__init__()

    def forward(self, log_pi):
        """
        forward pass
        :param log_pi: NxM tensor of log probabilities where N is the batch size and M is the number of classes
        :return: NxM tensor of 'discretized probabilities' the lowe the temperature the more discrete
        """
        pi = F.softmax(log_pi, 1)
        #z = F.sigmoid((pi-.5)*10000)

        return pi



class SpikeAndExp(pl.LightningModule):
    """
    Spike-and-exponential smoother from the original DVAE paper of Rolfe.
    """
    def __init__(self, **kwargs):
        super(SpikeAndExp, self).__init__()
        beta = kwargs.get('beta', None)
        self.beta = torch.Tensor([beta]).float()

    def forward(self, q):
        #clip the probabilities
        q = torch.clamp(q,min=1e-7,max=1.-1e-7)

        #this is a tensor of uniformly sampled random number in [0,1)
        rho = torch.rand(q.size()).to(q)
        zero_mask = torch.zeros(q.size()).to(q)
        ones = torch.ones(q.size()).to(q)
        beta = self.beta.to(q)


        # inverse CDF

        conditional_log = (1./beta)*torch.log(((rho+q-ones)/q)*(beta.exp()-1)+ones)

        zeta=torch.where(rho >= 1 - q, conditional_log, zero_mask)
        return zeta

class STEFunction(torch.autograd.Function):
    """
    Class for straight through estimator. Samples from a tensor of proabilities, but defines gradient as if
    forward is an identity()
    """
    @staticmethod
    def forward(ctx, probs):
        """
        Sample from multinomial distribution given a tensor of probabiltieis
        :param ctx: unused argument for passing class
        :param probs: NxM tensor of probabilties
        :return: output: NxM binary tensor containing smapled values
        """
        sample_ix = torch.multinomial(probs, 1).squeeze()
        output = torch.zeros_like(probs)
        output[torch.arange(output.shape[0]), sample_ix] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        gradients for straight trough estimator
        :param ctx: unused argument for passing class
        :param grad_output: the gradients passed to this function
        :return: simply the gradients clamped to be fbetween -1 and 1
        """
        return F.hardtanh(grad_output)


class StraightThroughSampler(nn.Module):
    """
    Neural network module for straight though sampler. Samples a value in forward but returns gradients as
    if there was no sampling
    """
    def __init__(self):
        super(StraightThroughSampler, self).__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x


class MultinomialSampler(nn.Module):
    """
    Neural network module for multinomial sampler.
    """
    def __init__(self):
        super(MultinomialSampler, self).__init__()

    def forward(self, probs):
        sample_ix =  torch.multinomial(probs, 1).squeeze()
        one_hot = torch.nn.functional.one_hot(sample_ix)

        return one_hot.float()


class VectorQuantizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        n_emb = kwargs.get('nclass', None)
        emb_dim = kwargs.get('emb_dim', None)
        self.embeddings = nn.Embedding(n_emb, emb_dim)
        self.vq_loss = 0

    def forward(self, ze):
        # compute the distances between the encoder outputs and the embeddings
        dist_mat = torch.cdist(ze, self.embeddings.weight.clone(), p=2)
        # select closest embedding for each person
        emb_ix = torch.argmin(dist_mat, dim=2)

        zq = self.embeddings(emb_ix)
        #ze = ze.squeeze()
        self.vq_loss = F.mse_loss(zq.detach(), ze) + F.mse_loss(zq, ze.detach())

        zq = ze + (zq-ze).detach()
        return zq

    def closest_emb_ix(self, ze):
        # compute the distances between the encoder outputs and the embeddings
        dist_mat = torch.cdist(ze, self.embeddings.weight.clone(), p=2)
        # return closest embedding for each person
        return torch.argmin(dist_mat, dim=1)


