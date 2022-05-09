import time
import numpy as np

from numpy.random import randint
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans

from agent import Agent


class Executor:
    measure_map = {
        'silhouette': (silhouette_score, True),
        'davies_bouldin': (davies_bouldin_score, False),
        'calinski_harabasz': (calinski_harabasz_score, True)
    }

    def __init__(self, ds, stat, mutator, measure, *, attempts=70, init='random', gamma=0.95, lam=8, amount=4, max_cl=25):
        self.ds, self.stat, self.mutator, self.max_cl = ds, stat, mutator, max_cl
        self.init_partition = self.kmeans_partition if init == 'k-means' else self.random_partition
        self.measure, self.inc = self.measure_map[measure]
        self.attempts, self.gamma, self.lam, self.amount = attempts, gamma, lam, amount

    def score(self, labels):
        return self.measure(self.ds, labels) * (1.0 if self.inc else -1.0)

    def random_partition(self, k):
        clusters = [list() for _ in range(k)]
        for p in range(self.ds.shape[0]):
            clusters[randint(0, k)].append(p)
        return list(filter(lambda cl: len(cl) > 0, clusters))

    def kmeans_partition(self, k):
        labels = KMeans(k, init='k-means++', n_init=1, max_iter=1).fit_predict(self.ds)
        return self.stat.split_by_clusters(labels)

    def init_single(self):
        partition = self.init_partition(randint(2, self.max_cl))
        clusters = self.mutator.clean_clusters(partition)
        if clusters is None:
            return self.init_single()
        labs = self.stat.unite_by_labels(clusters)
        return labs, self.score(labs)

    def initialise(self):
        initials = [self.init_single() for _ in range(self.lam)]
        labs, fits = zip(*initials)
        return list(labs), list(fits)

    def single_step(self, agent, metrics, labels):
        policy, samples = agent.calc_policy(metrics), list()
        arms = np.random.choice(len(policy), size=self.amount, replace=False, p=policy)
        new_clusters = [self.mutator(labels, arm) for arm in arms]
        for arm_idx, partition in enumerate(new_clusters):
            if partition is not None:
                new_labs = self.stat.unite_by_labels(partition)
                new_metrics = self.stat.calc_metrics(new_labs)
                samples.append((arms[arm_idx], new_labs, new_metrics, self.score(new_labs)))
        if len(samples) == 0:
            return policy, None, None, None, None
        samples.sort(key=lambda s: s[0])
        used_arms, labs, new_inputs, fits = zip(*samples)
        return policy, list(labs), np.array(new_inputs), np.array(fits), np.array(used_arms)

    def make_step(self, agent, inputs, labels, fitness):
        used_arms = list()
        for idx, metrics in enumerate(inputs):
            policy, labs, new_metrics, fits, arms = self.single_step(agent, metrics, labels[idx])
            if arms is not None:
                agent.backprop(metrics, new_metrics, arms, fits - fitness[idx])
                next_gen = np.argmax(fits)
                labels[idx], fitness[idx], inputs[idx] = labs[next_gen], fits[next_gen], new_metrics[next_gen]
                used_arms.extend(arms)
            else:
                labels[idx], fitness[idx] = self.init_single()
                inputs[idx] = self.stat.calc_metrics(labels[idx])
        return used_arms

    def launch(self, stamps):
        agent = Agent(len(self.mutator.mutations), self.gamma, self.lam)
        start, cur_stamp, epoch = time.time(), 0, 0
        results, best_fit, best_labs = list(), float('-inf'), None
        while True:
            labels, fitness = self.initialise()
            best_fit = max(best_fit, *fitness)
            input_metrics = [self.stat.calc_metrics(labs) for labs in labels]
            arms_usage = np.zeros(len(self.mutator.mutations), dtype=int)
            epoch_best, epoch = max(*fitness), epoch + 1
            for attempt in range(self.attempts):
                used_arms = self.make_step(agent, input_metrics, labels, fitness)
                for arm in used_arms:
                    arms_usage[arm] += 1
                cur_best = np.argmax(fitness)
                if fitness[cur_best] > best_fit:
                    best_fit, best_labs = fitness[cur_best], labels[cur_best]
                epoch_best = max(epoch_best, fitness[cur_best])
                if time.time() - start >= stamps[cur_stamp]:
                    print(f'{int(stamps[cur_stamp])}s: {best_fit}')
                    cur_stamp += 1
                    results.append(best_fit)
                    if cur_stamp == len(stamps):
                        return results, best_labs, best_fit
                print(', '.join('{:.3f}'.format(f) for f in fitness))
            print(f"=== BEST FOR EPOCH #{epoch} is {epoch_best} ===")
            print(arms_usage)
