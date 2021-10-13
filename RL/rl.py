import time
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from .config import get_config
from .actor import Actor
from .critic import Critic
from .Reward_BIC import get_Reward,BIC_lambdas
from datasets.dataloader import DataGenerator
from utils import utils
import matplotlib.pyplot as plt
from evaluation.evaluation import MetricsDAG

class RL(nn.Module):
    """
    RL Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/1906.04477

    Examples
    --------
    """

    def __init__(self):
        super().__init__()

    def learn(self, data, true_dag, **kwargs):
        """
        Set up and run the RL algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        """
        config, _ = get_config()
        for k in kwargs:
            config.__dict__[k] = kwargs[k]

        if isinstance(data, np.ndarray):
            X = data

        else:
            raise TypeError('The type of data must be '
                            'Tensor or numpy.ndarray, but got {}'
                            .format(type(data)))

        config.data_size = X.shape[0]
        config.max_length = X.shape[1]
        self.config = config

        causal_matrix = self._rl(X,true_dag, config)
        self.causal_matrix = causal_matrix

    def _rl(self, X, true_dag, config):

        # input data
        if hasattr(config, 'dag'):
            training_set = DataGenerator(
                X, config.dag, config.normalize, config.transpose)
        else:
            training_set = DataGenerator(
                X, None, config.normalize, config.transpose)

        # set penalty weights
        score_type = config.score_type
        reg_type = config.reg_type

        sl, su, strue = BIC_lambdas(training_set.inputdata, None, None, None, reg_type, score_type)
        lambda1 = 0
        lambda1_upper = 5
        lambda1_update_add = 1
        lambda2 = 1/(10**(np.round(config.max_length/3)))
        lambda2_upper = 0.01
        lambda2_update_mul = 10
        lambda_iter_num = config.lambda_iter_num

        # test initialized score
        logger.info('Original sl: {}, su: {}, strue: {}'.format(sl, su, strue))
        logger.info('Transfomed sl: {}, su: {}, lambda2: {}, true: {}'.format(sl, su, lambda2,
                                                                              (strue-sl)/(su-sl)*lambda1_upper))

        # actor
        actor = Actor(config).cuda(config.gpu)
        critic = Critic(config).cuda(config.gpu)

        callreward = get_Reward(batch_num=actor.batch_size, maxlen=config.max_length,
                                dim=actor.input_dimension, inputdata=training_set.inputdata,
                                sl=sl, su=su, lambda1_upper=lambda1_upper, score_type=score_type, reg_type=reg_type,
                                l1_graph_reg=config.l1_graph_reg, verbose_flag=False)
        logger.info('Finished creating training dataset, actor model and reward class')
        logger.info('Starting session...')

        # Run initialize op
        # Test tensor shape

        # Initialize useful variables
        rewards_avg_baseline = []
        rewards_batches = []
        reward_max_per_batch = []

        lambda1s = []
        lambda2s = []

        graphss = []
        probsss = []
        max_rewards = []
        max_reward = float('-inf')
        max_reward_score_cyc = (lambda1_upper+1, 0)

        logger.info('Starting training.')

        for i in (range(1, config.nb_epoch + 1)):

            if config.verbose:
                logger.info('Start training for {}-th epoch'.format(i))
            input_batch = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
            input_batch = torch.tensor(input_batch).to(torch.float32).cuda(config.gpu)
            encoder_output, graphs_feed2,_,_ = actor(input_batch)
            graphs_feed = graphs_feed2.cpu().detach().numpy()
            torch.cuda.empty_cache()
            reward_feed = callreward.cal_rewards(graphs_feed, lambda1, lambda2)

            # max reward, max reward per batch
            max_reward = -callreward.update_scores([max_reward_score_cyc], lambda1, lambda2)[0]
            max_reward_batch = float('inf')
            max_reward_batch_score_cyc = (0, 0)

            for reward_, score_, cyc_ in reward_feed:
                if reward_ < max_reward_batch:
                    max_reward_batch = reward_
                    max_reward_batch_score_cyc = (score_, cyc_)

            max_reward_batch = -max_reward_batch

            if max_reward < max_reward_batch: #batch中最小的reward是最大的
                max_reward = max_reward_batch
                max_reward_score_cyc = max_reward_batch_score_cyc

            # for average reward per batch
            reward_batch_score_cyc = np.mean(reward_feed[:,1:], axis=0)

            if config.verbose:
                logger.info('Finish calculating reward for current batch of graph')

            prediction = critic(encoder_output)
            actor.decoder.buff()
            reward_avg_baseline = actor.build_optim(-reward_feed[:,0], prediction)
            reward_batch = np.mean(-reward_feed[:,0], axis= 0)
            encoder_output,_,graph_batch,probs = actor(input_batch)
            reward = torch.from_numpy(-reward_feed[:,0]).float().cuda(self.config.gpu)
            avg_baseline = torch.tensor(actor.avg_baseline).float().cuda(self.config.gpu)
            critic.optimizer(reward, avg_baseline, encoder_output=encoder_output)
            torch.cuda.empty_cache()

            if config.verbose:
                logger.info('Finish updating actor and critic network using reward calculated')

            lambda1s.append(lambda1)
            lambda2s.append(lambda2)

            rewards_avg_baseline.append(reward_avg_baseline)
            rewards_batches.append(reward_batch_score_cyc)
            reward_max_per_batch.append(max_reward_batch_score_cyc)
            max_rewards.append(max_reward_score_cyc)
            actor.decoder.buff()

            # logging
            if i == 1 or i % 500 == 0:
                logger.info('[iter {}] reward_batch: {:.4}, max_reward: {:.4}, max_reward_batch: {:.4}'.format(i,reward_batch, max_reward, max_reward_batch))

            # update lambda1, lamda2
            if i == 1 or i % lambda_iter_num == 0:
                ls_kv = callreward.update_all_scores(lambda1, lambda2)

                graph_int, score_min, cyc_min = np.int32(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]

                if cyc_min < 1e-5:
                    lambda1_upper = score_min
                lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
                lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
                logger.info('[iter {}] lambda1 {:.4}, upper {:.4}, lambda2 {:.4}, upper {:.4}, score_min {:.4}, cyc_min {:.4}'.format(i,
                          lambda1*1.0, lambda1_upper*1.0, lambda2*1.0, lambda2_upper*1.0, score_min*1.0, cyc_min*1.0))
                graph_batch = utils.convert_graph_int_to_adj_mat(graph_int)

                if reg_type == 'LR':
                    graph_batch_pruned = np.array(utils.graph_prunned_by_coef(graph_batch, training_set.inputdata))
                elif reg_type == 'QR':
                    graph_batch_pruned = np.array(utils.graph_prunned_by_coef_2nd(graph_batch, training_set.inputdata))

                # elif reg_type == 'GPR':
                #     # The R codes of CAM pruning operates the graph form that (i,j)=1 indicates i-th node-> j-th node
                #     # so we need to do a tranpose on the input graph and another tranpose on the output graph
                #     graph_batch_pruned = np.transpose(pruning_cam(training_set.inputdata, np.array(graph_batch).T))

                # estimate accuracy
                #if hasattr(config, 'dag'):
                met = MetricsDAG(graph_batch.T, true_dag)
                met2 = MetricsDAG(graph_batch_pruned.T, true_dag)
                acc_est = met.metrics
                acc_est2 = met2.metrics
                print(acc_est2)

                fdr, tpr, fpr, shd, nnz = \
                    acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], \
                    acc_est['shd'], acc_est['nnz']
                fdr2, tpr2, fpr2, shd2, nnz2 = \
                    acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], \
                    acc_est2['shd'], acc_est2['nnz']

                logger.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
                logger.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))

        plt.figure(1)
        plt.plot(rewards_batches, label='reward per batch')
        plt.plot(max_rewards, label='max reward')
        plt.legend()
        plt.savefig('reward_batch_average.png')
        plt.show()
        plt.close()

        logger.info('Training COMPLETED !')

        return graph_batch_pruned.T