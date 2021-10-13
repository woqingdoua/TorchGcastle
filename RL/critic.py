import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):

    def __init__(self, config):
        super(Critic, self).__init__()
        self.config=config

        # Data config
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.input_dimension = config.input_dimension
        self.lr2 = config.lr1_start

        # Network config
        self.input_embed = config.hidden_dim
        self.num_neurons = config.hidden_dim

        # Baseline setup
        self.init_baseline = 0.
        self.global_step = 0

        self.ffn1 = nn.Sequential(nn.Linear(self.input_embed, self.num_neurons), nn.ReLU())
        self.w1 = nn.Parameter(data=torch.rand(self.num_neurons, 1), requires_grad=True)
        self.b1 = nn.Parameter(data=torch.rand(self.num_neurons), requires_grad=True)

        self.opt2 = torch.optim.Adam(self.parameters(), lr=self.lr2, betas=(0.9, 0.99), eps=0.0000001)
        # Training config (critic)  # global step
        self.lr2_start = config.lr1_start  # initial learning rate
        self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt2, self.lr2_decay_rate)

    def forward(self, encoder_output):

        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = torch.sum(encoder_output, 1)

        # ffn 1
        h0 = self.ffn1(frame)
        # ffn 2
        predictions = torch.matmul(h0, self.w1).squeeze() + self.b1
        return predictions

    def optimizer(self,reward,avg_baseline,encoder_output):

        self.global_step = self.global_step + 1
        self.opt2.zero_grad()
        loss2 = F.mse_loss(self(encoder_output),reward - avg_baseline)

        # Minimize step
        loss2.backward()
        nn.utils.clip_grad_norm_(self.parameters(),1)
        self.opt2.step()
        if self.global_step % self.lr2_decay_step ==0:
            self.scheduler.step()