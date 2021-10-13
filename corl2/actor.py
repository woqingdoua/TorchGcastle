import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self,config):
        super(Actor, self).__init__()

        self.is_train = True
        self.config = config
        self.batch_size = config.batch_size
        self.avg_baseline = config.init_baseline
        self.input_dimension = config.input_dimension
        self.max_length = config.max_length
        self.alpha = config.alpha  #
        self.encoder = TransformerEncoder(self.config, self.is_train)
        self.decoder = Pointer_decoder(self.config)
        self.lr1 = 0.0001
        self.opt1 = torch.optim.Adam(self.parameters(), lr=self.lr1, betas=(0.9, 0.99), eps=0.0000001)

    def forward(self,inputs, train=True):

        encoder_output = self.encoder.encode(inputs)
        if train == False:
            return encoder_output
        positions, mask_scores, s0_list,s1_list, i_list = self.decoder.loop_decode(encoder_output)
        return encoder_output,positions, mask_scores, s0_list,s1_list, i_list

    def action(self,encoder_output,prev_state_0, prev_state_1, prev_input, position, action_mask_):

        log_softmax = self.decoder.decode_softmax(encoder_output,prev_state_0, prev_state_1, prev_input, position, action_mask_)
        self.log_softmax_ = torch.permute(log_softmax.view(self.batch_size, self.max_length),(1,0))
        self.log_softmax = torch.sum(self.log_softmax_, 0)  # TODO:[Batch,]


    def build_optim(self,reward,predictions):

        #baseline
        reward_mean = np.mean(reward, axis=0)
        self.reward_batch = reward_mean
        self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean

        # Actor learning rate,'reinforce'
        self.opt1.zero_grad()
        a = torch.from_numpy(reward - self.avg_baseline).cuda(self.config.gpu)
        self.reward_baseline =  a - predictions  # [Batch size, 1]
        loss1 = - torch.mean(self.reward_baseline * self.log_softmax, 0)
        loss1.backward()
        self.opt1.step()

        return loss1

class TransformerEncoder(nn.Module):

    def __init__(self, config, is_train):
        super(TransformerEncoder, self).__init__()
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.input_dimension # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token

        self.input_embed = config.hidden_dim # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks

        self.is_training = is_train #not config.inference_mode
        self.multihead_attention = multihead_attention(inputs=config.hidden_dim,num_units=self.input_embed, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)
        self.feedforward = feedforward(inputs=config.hidden_dim, num_units=[4*self.input_embed, self.input_embed], is_training=self.is_training)

        self.W_embed = torch.nn.Conv1d(64,64,kernel_size=1)
        self.batch_norm = torch.nn.BatchNorm1d(64)

    def encode(self, inputs):

        embedded_input = self.W_embed(inputs.permute(0,2,1))
        enc = self.batch_norm(embedded_input).permute(0,2,1)

        for i in range(self.num_stacks):
            enc = self.multihead_attention(enc)
            enc = self.feedforward(enc)

        return enc

class multihead_attention(nn.Module):
    def __init__(self,inputs, num_units=None, num_heads=16, dropout_rate=0., is_training=True):
        super(multihead_attention, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.is_training = is_training

        # Linear projections
        self.Q = nn.Sequential(nn.Linear(inputs, num_units),nn.ReLU()) # [batch_size, seq_length, n_hidden]
        self.K = nn.Sequential(nn.Linear(inputs, num_units),nn.ReLU())  # [batch_size, seq_length, n_hidden]
        self.V = nn.Sequential(nn.Linear(inputs, num_units),nn.ReLU()) # [batch_size, seq_length, n_hidden]
        self.batch_norm = torch.nn.BatchNorm1d(64)

    def forward(self,input):

        # Split and concat
        Q = self.Q(input)
        K = self.K(input)
        V = self.V(input)
        num_heads = Q.size()[-1]//self.num_heads
        Q_ = torch.cat(torch.split(Q, num_heads, dim=2), dim=0) # [batch_size*num_heads, seq_length, n_hidden/num_heads]
        K_ = torch.cat(torch.split(K, num_heads, dim=2), dim=0) # [batch_size, seq_length, n_hidden/num_heads]
        V_ = torch.cat(torch.split(V, num_heads, dim=2), dim=0) # [batch_size, seq_length, n_hidden/num_heads]

        # Multiplication
        outputs = torch.matmul(Q_, K_.permute(0,2,1)) # [num_heads*batch_size, seq_length, seq_length]

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        outputs = F.softmax(outputs,dim=-1) # [num_heads*batch_size, seq_length, seq_length]

        # Dropouts
        outputs = F.dropout(outputs, p=self.dropout_rate, training=self.is_training)

        # Weighted sum
        outputs = torch.matmul(outputs, V_) # [num_heads*batch_size, seq_length, n_hidden/num_heads]

        # Restore shape
        outputs = torch.cat(torch.split(outputs,  outputs.size()[0]//self.num_heads, dim=0), dim=2) # [batch_size, seq_length, n_hidden]

        # Residual connection
        outputs += input # [batch_size, seq_length, n_hidden]

        # Normalize
        outputs = self.batch_norm(outputs.permute(0,2,1)).permute(0,2,1)  # [batch_size, seq_length, n_hidden]

        return outputs

class feedforward(nn.Module):
    def __init__(self,inputs, num_units=[2048, 512], is_training=True):
        super(feedforward, self).__init__()

        self.is_training = is_training
        self.inner_layer = nn.Sequential(nn.Conv1d(inputs,num_units[0],kernel_size=1),nn.ReLU())
        self.readout_layer = nn.Sequential(nn.Conv1d(num_units[0],num_units[1],kernel_size=1),nn.ReLU())
        self.batch_norm = torch.nn.BatchNorm1d(64)

    def forward(self,inputs):
        inputs = inputs.permute(0,2,1)
        outputs = self.inner_layer(inputs)
        outputs = self.readout_layer(outputs)
        outputs += inputs
        outputs = self.batch_norm(outputs).permute(0,2,1)

        return outputs

class Pointer_decoder(nn.Module):
    '''
    RNN decoder for pointer network
    '''

    def __init__(self, config):
        super(Pointer_decoder, self).__init__()

        self.seq_length = config.max_length # sequence length
        self.n_hidden = config.hidden_dim # num_neurons

        self.C = config.C # logit clip

        self.buff()

        self.first_state = nn.Parameter(data=torch.Tensor(1,self.n_hidden), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.first_state)

        # Attending mechanism
        self.W_ref_g = nn.Conv1d(self.n_hidden,self.n_hidden,kernel_size=1)
        torch.nn.init.xavier_uniform_(self.W_ref_g.weight)
        self.W_q_g = nn.Parameter(data=torch.Tensor(self.n_hidden,self.n_hidden), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W_q_g)
        self.v_g = nn.Parameter(data=torch.rand(self.n_hidden), requires_grad=True)

        # Pointing mechanism
        self.W_ref = nn.Conv1d(self.n_hidden,self.n_hidden,kernel_size=1)
        torch.nn.init.xavier_uniform_(self.W_ref.weight)
        self.W_q = nn.Parameter(data=torch.Tensor(self.n_hidden,self.n_hidden), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W_q)
        self.v = nn.Parameter(data=torch.rand(self.n_hidden), requires_grad=True)

        # Decoder LSTM cell
        self.cell = nn.LSTM(self.n_hidden, self.n_hidden,batch_first=True)

    # From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
    # predict a distribution over next decoder input
    def buff(self):

        self.log_softmax = [] # store log(p_theta(pi(t)|pi(<t),s)) for backprop
        self.positions = [] # store visited cities for reward

        self.mask = 0
        self.mask_scores = []

    def attention(self,ref,query):
        # Attending mechanism
        encoded_ref_g = self.W_ref_g(ref.permute(0,2,1)).permute(0,2,1) # [Batch size, seq_length, n_hidden]
        encoded_query_g = torch.matmul(query, self.W_q_g).unsqueeze(1) # [Batch size, 1, n_hidden]
        scores_g = torch.sum(self.v_g * torch.tanh(encoded_ref_g + encoded_query_g),dim=-1) # [Batch size, seq_length]
        attention_g = F.softmax(scores_g - 100000000.*self.mask, dim=-1)  ###########
        # 1 glimpse = Linear combination of reference vectors (defines new query vector)
        glimpse = torch.multiply(ref, attention_g.unsqueeze(2)) #64*12*64 64*12*1
        glimpse = torch.sum(glimpse,1)+query  ########### Residual connection

        # Pointing mechanism with 1 glimpse
        encoded_ref = self.W_ref(ref.permute(0,2,1)).permute(0,2,1) # [Batch size, seq_length, n_hidden]
        encoded_query = torch.matmul(glimpse.float(), self.W_q).unsqueeze(1) # [Batch size, 1, n_hidden]
        scores = torch.sum(self.v * torch.tanh(encoded_ref + encoded_query), dim=-1) # [Batch size, seq_length]
        masked_scores = scores - 10000000.*self.mask # [Batch size, seq_length]

        return masked_scores

    def decode_softmax(self,encoder_output, prev_state_0, prev_state_1, prev_input, position, mask):

        s = prev_state_0.unsqueeze(0), prev_state_1.unsqueeze(0)

        output, state = self.cell(prev_input.unsqueeze(1), s)
        output = output.squeeze()
        self.mask = mask
        encoder_output_ex = torch.reshape(torch.tile((encoder_output.unsqueeze(1)),[1,self.seq_length, 1,1]),
                                          [-1, self.seq_length, self.n_hidden])
        masked_scores = self.attention(encoder_output_ex, output)  # [batch_size, time_sequence]
        prob = torch.distributions.categorical.Categorical(F.gumbel_softmax(masked_scores,tau=3))
        log_softmax = prob.log_prob(position)

        return log_softmax

    def decode(self,prev_state,prev_input,encoder_output):

        # Run the cell on a combination of the previous input and state
        prev_state = prev_state[0].unsqueeze(0), prev_state[1].unsqueeze(0)
        output, state = self.cell(prev_input.unsqueeze(1),prev_state)
        output = output.squeeze()
        state = state[0].squeeze(),state[1].squeeze()
        masked_scores = self.attention(encoder_output, output)  # [batch_size, time_sequence]
        self.mask_scores.append(masked_scores)

        # Multinomial distribution
        prob = torch.distributions.categorical.Categorical(F.gumbel_softmax(masked_scores, tau=3))
        # Sample from distribution
        position = prob.sample()
        self.positions.append(position)

        # Store log_prob for backprop
        self.log_softmax.append(prob.log_prob(position))

        self.mask = self.mask + F.one_hot(position, self.seq_length)

        # Retrieve decoder's new input
        h = encoder_output.permute(1, 0, 2)
        new_decoder_input = h.gather(0, position.unsqueeze(1).unsqueeze(1).repeat(1,len(position),len(position)))[0] #(64,64,64) (64,64)

        return state, new_decoder_input

    def loop_decode(self,encoder_output):
        # decoder_initial_state: Tuple Tensor (c,h) of size [batch_size x cell.state_size]
        # decoder_first_input: Tensor [batch_size x cell.state_size]

        # Loop the decoding process and collect results
        # Decoder initial state (tuple) is trainable
        decoder_first_input = torch.sum(encoder_output, 1)
        decoder_initial_state = torch.tile(self.first_state, (len(encoder_output),1)), decoder_first_input
        s, i = decoder_initial_state, decoder_first_input
        s0_list = []
        s1_list = []
        i_list = []
        for step in range(self.seq_length):
            s0_list.append(s[0])
            s1_list.append(s[1])
            i_list.append(i)
            s, i = self.decode(s, i, encoder_output)

        self.s0_list = torch.stack(s0_list, dim=1)  # [Batch,seq_length,hidden]
        self.s1_list = torch.stack(s1_list, dim=1)  # [Batch,seq_length,hidden]
        self.i_list = torch.stack(i_list, dim=1)  # [Batch,seq_length,hidden]

        # Stack visited indices
        self.positions = torch.stack(self.positions, dim=1)  # [Batch,seq_length]
        self.mask_scores = torch.stack(self.mask_scores, dim=1)  # [Batch,seq_length,seq_length]
        self.log_softmax = torch.stack(self.log_softmax, dim=0)
        #self.log_softmax_first_run = tf.add_n(self.log_softmax_first_run)  # TODO:[Batch,]
        self.mask = 0
        return self.positions, self.mask_scores, self.s0_list, self.s1_list, self.i_list

