from evaluation.plotdag import GraphDAG
from corl2.corl2 import CORL2
from rl.rl import RL
from evaluation.evaluation import MetricsDAG
from datasets.datasets import DAG, IIDSimulation


if __name__ == '__main__':

    model = RL() #RL()

    weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=20, weight_range=(0.5, 2.0), seed=1)
    dataset = IIDSimulation(W=weighted_random_dag, n=2000, method='linear', sem_type='gauss')
    true_dag, X = dataset.B, dataset.X
    print(true_dag)
    # rl learn
    model.learn(X,true_dag)

    # plot est_dag and true_dag
    GraphDAG(model.causal_matrix, true_dag)

    # calculate accuracy
    met = MetricsDAG(model.causal_matrix, true_dag)
    print(met.metrics)

