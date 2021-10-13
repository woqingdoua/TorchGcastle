import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):

    def __init__(self, config, is_train):
        super(Critic, self).__init__()
        self.config=config

        # Data config
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.input_dimension = config.input_dimension
        self.lr2 = 0.001

        # Network config
        self.input_embed = config.hidden_dim
        self.num_neurons = config.hidden_dim

        # Baseline setup
        self.init_baseline = 0.
        self.is_train = is_train

        self.ffn1 = nn.Sequential(nn.Linear(self.input_embed, self.num_neurons), nn.ReLU())
        self.w1 = nn.Parameter(data=torch.rand(self.num_neurons, 1), requires_grad=True)
        self.b1 = nn.Parameter(data=torch.rand(self.num_neurons), requires_grad=True)

        self.opt2 = torch.optim.Adam(self.parameters(), lr=self.lr2, betas=(0.9, 0.99), eps=0.0000001)

    def predict_rewards(self, encoder_output):

        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = torch.sum(encoder_output, 1)

        # ffn 1
        h0 = self.ffn1(frame)
        # ffn 2
        predictions = torch.matmul(h0, self.w1).squeeze() + self.b1
        return predictions

    def optimier(self,reward, avg_baseline, encoder_output):

        # Optimizer
        self.opt2.zero_grad()
        a = (torch.from_numpy(reward).float()-avg_baseline).cuda(self.config.gpu)
        loss2 = F.mse_loss(self.predict_rewards(encoder_output), a)

        # Minimize step
        loss2.backward()
        self.opt2.step()

        return loss2