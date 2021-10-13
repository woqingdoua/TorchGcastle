import logging
import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import expm as matrix_exponential
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

class get_Reward(object):

    _logger = logging.getLogger(__name__)

    def __init__(self, batch_num, maxlen, dim, inputdata, sl, su, lambda1_upper,
                 score_type='BIC', reg_type='LR', l1_graph_reg=0.0, verbose_flag=True):
        self.batch_num = batch_num
        self.maxlen = maxlen # =d: number of vars
        self.dim = dim
        self.baseint = 2**maxlen
        self.d = {} # store results
        self.d_RSS = {} # store RSS for reuse
        self.inputdata = inputdata
        self.n_samples = inputdata.shape[0]
        self.l1_graph_reg = l1_graph_reg
        self.verbose = verbose_flag
        self.sl = sl
        self.su = su
        self.lambda1_upper = lambda1_upper
        self.bic_penalty = np.log(inputdata.shape[0])/inputdata.shape[0]

        if score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')
        if reg_type not in ('LR', 'QR', 'GPR'):
            raise ValueError('Reg type not supported')
        self.score_type = score_type
        self.reg_type = reg_type

        self.ones = np.ones((inputdata.shape[0], 1), dtype=np.float32)
        self.poly = PolynomialFeatures()

    def cal_rewards(self, graphs, lambda1, lambda2):
        rewards_batches = []

        for graphi in graphs:
            reward_ = self.calculate_reward_single_graph(graphi, lambda1, lambda2)
            rewards_batches.append(reward_)

        return np.array(rewards_batches)

    ####### regression

    def calculate_yerr(self, X_train, y_train):
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train)
        elif self.reg_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.reg_type == 'GPR':
            return self.calculate_GPR(X_train, y_train)
        else:
            # raise value error
            assert False, 'Regressor not supported'

    # faster than LinearRegression() from sklearn
    def calculate_LR(self, X_train, y_train):
        X = np.hstack((X_train, self.ones))
        XtX = X.T.dot(X)
        Xty = X.T.dot(y_train)
        theta = np.linalg.solve(XtX, Xty)
        y_err = X.dot(theta) - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        X_train = self.poly.fit_transform(X_train)[:,1:]
        return self.calculate_LR(X_train, y_train)

    def calculate_GPR(self, X_train, y_train):
        med_w = np.median(pdist(X_train, 'euclidean'))
        gpr = GPR().fit(X_train/med_w, y_train)
        return y_train.reshape(-1,1) - gpr.predict(X_train/med_w).reshape(-1,1)

    ####### score calculations

    def calculate_reward_single_graph(self, graph_batch, lambda1, lambda2):
        graph_to_int = []
        graph_to_int2 = []

        for i in range(self.maxlen):
            graph_batch[i][i] = 0
            tt = np.int32(graph_batch[i])
            graph_to_int.append(self.baseint * i + np.int(''.join([str(ad) for ad in tt]), 2))
            graph_to_int2.append(np.int(''.join([str(ad) for ad in tt]), 2))

        graph_batch_to_tuple = tuple(graph_to_int2)

        if graph_batch_to_tuple in self.d:
            score_cyc = self.d[graph_batch_to_tuple]
            return self.penalized_score(score_cyc, lambda1, lambda2), score_cyc[0], score_cyc[1]

        RSS_ls = []

        for i in range(self.maxlen):
            col = graph_batch[i]
            if graph_to_int[i] in self.d_RSS:
                RSS_ls.append(self.d_RSS[graph_to_int[i]])
                continue

            # no parents, then simply use mean
            if np.sum(col) < 0.1:
                y_err = self.inputdata[:, i]
                y_err = y_err - np.mean(y_err)

            else:
                cols_TrueFalse = col > 0.5
                X_train = self.inputdata[:, cols_TrueFalse]
                y_train = self.inputdata[:, i]
                y_err = self.calculate_yerr(X_train, y_train)

            RSSi = np.sum(np.square(y_err))

            # if the regresors include the true parents, GPR would result in very samll values, e.g., 10^-13
            # so we add 1.0, which does not affect the monotoniticy of the score
            if self.reg_type == 'GPR':
                RSSi += 1.0

            RSS_ls.append(RSSi)
            self.d_RSS[graph_to_int[i]] = RSSi

        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls)/self.n_samples+1e-8) \
                  + np.sum(graph_batch)*self.bic_penalty/self.maxlen
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls)/self.n_samples+1e-8)) \
                  + np.sum(graph_batch)*self.bic_penalty

        score = self.score_transform(BIC)
        cycness = np.trace(matrix_exponential(np.array(graph_batch)))- self.maxlen
        reward = score + lambda1*np.float(cycness>1e-5) + lambda2*cycness

        if self.l1_graph_reg > 0:
            reward = reward + self.l1_grapha_reg * np.sum(graph_batch)
            score = score + self.l1_grapha_reg * np.sum(graph_batch)

        self.d[graph_batch_to_tuple] = (score, cycness)

        if self.verbose:
            self._logger.info('BIC: {}, cycness: {}, returned reward: {}'.format(BIC, cycness, score))

        return reward, score, cycness

    #### helper

    def score_transform(self, s):
        return (s-self.sl)/(self.su-self.sl)*self.lambda1_upper

    def penalized_score(self, score_cyc, lambda1, lambda2):
        score, cyc = score_cyc
        return score + lambda1*np.float(cyc>1e-5) + lambda2*cyc

    def update_scores(self, score_cycs, lambda1, lambda2):
        ls = []
        for score_cyc in score_cycs:
            ls.append(self.penalized_score(score_cyc, lambda1, lambda2))
        return ls

    def update_all_scores(self, lambda1, lambda2):
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_cyc in score_cycs:
            ls.append((graph_int, (self.penalized_score(score_cyc, lambda1, lambda2), score_cyc[0], score_cyc[1])))
        return sorted(ls, key=lambda x: x[1][0])


def BIC_input_graph(X, g, reg_type='LR', score_type='BIC'):
    """cal BIC score for given graph"""

    RSS_ls = []

    n, d = X.shape

    if reg_type in ('LR', 'QR'):
        reg = LinearRegression()
    else:
        reg =GaussianProcessRegressor()

    poly = PolynomialFeatures()

    for i in range(d):
        y_ = X[:, [i]]
        inds_x = list(np.abs(g[i])>0.1)

        if np.sum(inds_x) < 0.1:
            y_pred = np.mean(y_)
        else:
            X_ = X[:, inds_x]
            if reg_type == 'QR':
                X_ = poly.fit_transform(X_)[:, 1:]
            elif reg_type == 'GPR':
                med_w = np.median(pdist(X_, 'euclidean'))
                X_ = X_ / med_w
            reg.fit(X_, y_)
            y_pred = reg.predict(X_)
        RSSi = np.sum(np.square(y_ - y_pred))

        if reg_type == 'GPR':
            RSS_ls.append(RSSi+1.0)
        else:
            RSS_ls.append(RSSi)

    if score_type == 'BIC':
        return np.log(np.sum(RSS_ls)/n+1e-8)
    elif score_type == 'BIC_different_var':
        return np.sum(np.log(np.array(RSS_ls)/n)+1e-8)


def BIC_lambdas(X, gl=None, gu=None, gtrue=None, reg_type='LR', score_type='BIC'):
    """
    :param X: dataset
    :param gl: input graph to get score lower bound
    :param gu: input graph to get score upper bound
    :param gtrue: input true graph
    :param reg_type:
    :param score_type:
    :return: score lower bound, score upper bound, true score (only for monitoring)
    """

    n, d = X.shape

    if score_type == 'BIC':
        bic_penalty = np.log(n) / (n*d)
    elif score_type == 'BIC_different_var':
        bic_penalty = np.log(n) / n

    # default gl for BIC score: complete graph (except digonals)
    if gl is None:
        g_ones= np.ones((d,d))
        for i in range(d):
            g_ones[i, i] = 0
        gl = g_ones

    # default gu for BIC score: empty graph
    if gu is None:
        gu = np.zeros((d, d))

    sl = BIC_input_graph(X, gl, reg_type, score_type)
    su = BIC_input_graph(X, gu, reg_type, score_type)

    if gtrue is None:
        strue = sl - 10
    else:
        print(BIC_input_graph(X, gtrue, reg_type, score_type))
        print(gtrue)
        print(bic_penalty)
        strue = BIC_input_graph(X, gtrue, reg_type, score_type) + np.sum(gtrue) * bic_penalty

    return sl, su, strue
