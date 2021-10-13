import time
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from .config import get_config
from .actor import Actor
from .critic import Critic
from .Reward_BIC import get_Reward
from datasets.dataloader import DataGenerator
from utils import utils
import matplotlib.pyplot as plt
from evaluation.evaluation import MetricsDAG

class CORL2(nn.Module):
    """
    CORL2 Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix.
    """

    def __init__(self):
        super().__init__()

    def learn(self, data,true_dag, **kwargs):
        """
        Set up and run the CORL2 algorithm.

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

        causal_matrix = self._rl(X,true_dag, config)
        self.causal_matrix = causal_matrix

    def _rl(self, X,true_dag, config):
        """
        Starting model of CORL2.

        Parameters
        ----------
        X: numpy.ndarray
            The numpy.ndarray format data you want to learn.
        config: dict
            The parameters dict for corl2.
        """

        # input data

        solution_path_mask = None
        if hasattr(config, 'dag'):
            training_set = DataGenerator(
                X, config.dag, solution_path_mask, config.normalize, config.transpose)
        else:
            training_set = DataGenerator(
                X, solution_path_mask, config.normalize, config.transpose)
        input_data = training_set.inputdata[:config.data_size, :]

        # set penalty weights
        score_type = config.score_type
        reg_type = config.reg_type

        actor = Actor(config).cuda(config.gpu)
        critic = Critic(config, is_train=True).cuda(config.gpu)
        callreward = get_Reward(actor.batch_size, config.max_length,
                                config.parral, actor.input_dimension,
                                input_data, score_type, reg_type,
                                config.gpr_alpha, config.med_w,
                                config.median_flag, config.l1_graph_reg, False)
        logger.info('Finished creating training dataset, actor model and reward class')

        logger.info('Starting session...')

        # Initialize useful variables
        rewards_batches = []
        reward_max_per_batch = []

        max_rewards = []
        max_reward = float('-inf')

        max_sum = 0

        loss1_s, loss_2s = [], []
        beg_t = time.time()
        for i in (range(1, config.nb_epoch + 1)):

            if (time.time() - beg_t) > (15*3600):
                if i>1001:
                    break

            input_batch = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
            input_batch = torch.tensor(input_batch).to(torch.float32).cuda(config.gpu)

            encoder_output,positions, mask_scores, s0_list,s1_list, i_list = actor(input_batch)
            positions = positions.cpu()

            samples = []
            action_mask_s = []
            for m in range(positions.shape[0]):
                zero_matrix = utils.from_order_to_graph(positions[m])

                action_mask = np.zeros(actor.max_length)
                for po in positions[m]:
                    action_mask_s.append(action_mask.copy())
                    action_mask += np.eye(actor.max_length)[po]

                samples.append(zero_matrix)
                temp_sum = utils.cover_rate(zero_matrix, training_set.true_graph.T)
                if temp_sum > max_sum:
                    max_sum = temp_sum
                    if i == 1 or i % 500 == 0:
                        logger.info('[iter {}] [Batch {}_th] The optimal nodes order cover true graph {}/{}!'.format(i,m,max_sum,actor.max_length*actor.max_length))

            graphs_feed = np.stack(samples)
            action_mask_s = np.stack(action_mask_s)
            reward_feed = callreward.cal_rewards(graphs_feed, positions)

            max_reward_batch = -float('inf')
            reward_list, normal_batch_reward = [], []
            for nu, (reward_, reward_list_) in enumerate(reward_feed):
                reward_list.append(reward_list_)
                normalized_reward = -reward_
                normal_batch_reward.append(normalized_reward)
                if normalized_reward > max_reward_batch:
                    max_reward_batch = normalized_reward
            if max_reward < max_reward_batch:
                max_reward = max_reward_batch

            s0_list = s0_list.view(-1,config.hidden_dim)
            s1_list = s1_list.view(-1,config.hidden_dim)
            i_list = i_list.view(-1,config.hidden_dim)
            positions = positions.view(-1).cuda(config.gpu)
            action_mask_s = torch.from_numpy(action_mask_s).cuda(config.gpu)
            actor.action(encoder_output, s0_list, s1_list, i_list, positions, action_mask_s)
            normal_batch_reward = np.stack(normal_batch_reward)
            prediction = critic.predict_rewards(encoder_output)
            loss1 = actor.build_optim(normal_batch_reward,prediction)
            encoder_output = actor(input_batch, train=False)
            loss2 = critic.optimier(normal_batch_reward, actor.avg_baseline, encoder_output)

            loss1_s.append(loss1)
            loss_2s.append(loss2)
            reward_max_per_batch.append(max_reward_batch)
            rewards_batches.append(np.mean(normal_batch_reward))
            max_rewards.append(max_reward)
            actor.decoder.buff()
            # logging
            if i == 1 or i % 500 == 0:
                logger.info('[iter {}] reward_batch: {:.4}, max_reward: {:.4}, max_reward_batch: {:.4}'.format(i,
                        np.mean(normal_batch_reward), max_reward, max_reward_batch))

            if i == 1 or (i+1) % config.lambda_iter_num == 0:
                ls_kv = callreward.update_all_scores()

                score_min, graph_int_key = ls_kv[0][1][0], ls_kv[0][0]
                logger.info('[iter {}] score_min {:.4}'.format(i, score_min*1.0))
                graph_batch = utils.from_order_to_graph(graph_int_key)

                temp_sum = utils.cover_rate(graph_batch, training_set.true_graph.T)

                if reg_type == 'LR':
                    graph_batch_pruned = np.array(utils.graph_prunned_by_coef(graph_batch, input_data))
                elif reg_type == 'QR':
                    graph_batch_pruned = np.array(utils.graph_prunned_by_coef_2nd(graph_batch, input_data))
                # elif reg_type == 'GPR' or reg_type == 'GPR_learnable':
                #     graph_batch_pruned = np.transpose(pruning_cam(input_data,
                #                                                     np.array(graph_batch).T))

                if hasattr(config, 'dag'):
                    met = MetricsDAG(graph_batch.T, true_dag)
                    met2 = MetricsDAG(graph_batch_pruned.T, true_dag)
                    acc_est = met.metrics
                    acc_est2 = met2.metrics

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
        plt.plot(reward_max_per_batch, label='max reward per batch')
        plt.plot(max_rewards, label='max reward')
        plt.legend()
        plt.savefig('reward_batch_average.png')
        plt.show()
        plt.close()

        logger.info('Training COMPLETED!')
        return graph_batch_pruned.T