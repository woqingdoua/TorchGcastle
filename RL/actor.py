import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from corl2.actor import TransformerEncoder


class Actor(nn.Module):

    def __init__(self,config):
        super(Actor, self).__init__()

        self.config = config
        self.is_train = True
        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length
        self.input_dimension = config.input_dimension

        # Reward config
        self.avg_baseline = config.init_baseline  # moving baseline for Reinforce
        self.alpha = config.alpha  # moving average update

        # Training config (actor)
        self.global_step = 0  # global step
        self.lr1_start = config.lr1_start  # initial learning rate
        self.lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr1_decay_step = config.lr1_decay_step  # learning rate decay step

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.encoder = TransformerEncoder(self.config, self.is_train)
        self.decoder = SingleLayerDecoder(self.config, self.is_train)

        self.opt1 = torch.optim.Adam(self.parameters(), lr=self.lr1_start, betas=(0.9, 0.99), eps=0.0000001)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt1, self.lr1_decay_rate)

    def forward(self, input_):
        encoder_output = self.encoder.encode(input_)
        self.samples, self.scores, self.entropy = self.decoder.decode(encoder_output)

        # self.samples is seq_lenthg * batch size * seq_length
        # cal cross entropy loss * reward
        graphs_gen = torch.stack(self.samples).permute(1,0,2)
        self.graphs = graphs_gen
        self.graph_batch = torch.mean(self.graphs, dim=0)
        logits_for_rewards = torch.stack(self.scores)
        entropy_for_rewards = torch.stack(self.entropy)
        entropy_for_rewards = entropy_for_rewards.permute(1, 0, 2)
        logits_for_rewards = logits_for_rewards.permute(1, 0, 2)
        self.log_softmax = F.binary_cross_entropy_with_logits(logits_for_rewards,self.graphs)
        self.entropy_regularization = torch.mean(torch.mean(entropy_for_rewards, dim=2),dim=1)

        return encoder_output, self.graphs, self.graph_batch, self.log_softmax


    def build_optim(self,reward,predictions,):

        #baseline
        reward_mean = np.mean(reward, axis=0)
        self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean

        # Discounted reward
        self.opt1.zero_grad()
        a = torch.from_numpy(reward - self.avg_baseline).cuda(self.config.gpu)
        self.reward_baseline = a - predictions  # [Batch size, 1]

        # Loss
        self.global_step = self.global_step + 1
        loss1 = torch.mean(self.reward_baseline * self.log_softmax, dim=0) - self.lr1_start*torch.mean(self.entropy_regularization, dim=0)
        loss1.backward()
        nn.utils.clip_grad_norm_(self.parameters(),1)
        self.opt1.step()
        if self.global_step % self.lr1_decay_step ==0:
            self.scheduler.step()
        self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean
        return self.avg_baseline,reward_mean

class SingleLayerDecoder(nn.Module):
    def __init__(self,config, is_train):
        super(SingleLayerDecoder, self).__init__()

        self.batch_size = config.batch_size    # batch size
        self.max_length = config.max_length    # input sequence length (number of cities)
        self.input_dimension = config.hidden_dim
        self.input_embed = config.hidden_dim    # dimension of embedding space (actor)
        self.max_length = config.max_length
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.decoder_activation = config.decoder_activation
        self.use_bias = config.use_bias
        self.bias_initial_value = config.bias_initial_value
        self.use_bias_constant = config.use_bias_constant
        self.config = config

        self.is_training = is_train

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []

        self.W_l = nn.Parameter(torch.rand(self.input_embed, self.decoder_hidden_dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W_l)
        self.W_r = nn.Parameter(torch.rand(self.input_embed, self.decoder_hidden_dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W_r)
        self.U = nn.Parameter(torch.rand(self.decoder_hidden_dim), requires_grad=True)    # Aggregate across decoder hidden dim
        self.logit_bias = nn.Parameter(torch.rand(1), requires_grad=True)

    def decode(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]

        dot_l = torch.einsum('ijk, kl->ijl', encoder_output, self.W_l)
        dot_r = torch.einsum('ijk, kl->ijl', encoder_output, self.W_r)

        tiled_l = torch.tile(dot_l.unsqueeze(2), (1, 1, self.max_length, 1))
        tiled_r = torch.tile(dot_r.unsqueeze(1), (1, self.max_length, 1, 1))

        final_sum = torch.tanh(tiled_l + tiled_r)
        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = torch.einsum('ijkl, l->ijk', final_sum, self.U)    # Readability

        if self.use_bias:    # Bias to control sparsity/density
            logits += self.logit_bias

        self.adj_prob = logits

        for i in range(self.max_length):
            position = torch.ones([encoder_output.shape[0]]) * i

            # Update mask
            self.mask = F.one_hot(position.long(), self.max_length).cuda(self.config.gpu)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = torch.distributions.bernoulli.Bernoulli(torch.sigmoid(masked_score))    # probs input probability, logit input log_probability

            sampled_arr = prob.sample()    # Batch_size, seqlenght for just one node

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy

    def buff(self):

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []