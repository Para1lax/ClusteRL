import itertools
import time
import numpy as np

from numpy.random import randint
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from agent import Agent


def d(x, y):
    return np.linalg.norm(x - y)


# custom implementation
def gD43(dataset, labels):
    k = np.max(labels) + 1
    clusters = [list() for _ in range(k)]
    for idx, sample in enumerate(labels):
        clusters[sample].append(idx)
    centroids, diams = [sum([dataset[idx] for idx in cl]) / len(cl) for cl in clusters], list()
    c_dists = [d(centroids[x], centroids[y]) for x, y in itertools.combinations(range(k), 2)]
    for lab, cluster in enumerate(clusters):
        diams.append(2.0 * np.mean([d(dataset[idx], centroids[lab]) for idx in cluster]))
    return min(c_dists) / max(diams)


class Executor:
    measure_map = {
        'silhouette': (silhouette_score, True),
        'calinski_harabasz': (calinski_harabasz_score, True),
        'davies_bouldin': (davies_bouldin_score, False),
        'gD43': (gD43, True)
    }

    def __init__(self, ds, stat, mutator, measure, *, attempts=64, gamma=0.95, lam=8, amount=4, max_cl=25):
        self.ds, self.stat, self.mutator, self.max_cl = ds, stat, mutator, max_cl
        self.measure, self.inc = self.measure_map[measure]
        self.attempts, self.gamma, self.lam, self.amount = attempts, gamma, lam, amount

    def score(self, labels):
        return self.measure(self.ds, labels) * (1.0 if self.inc else -1.0)

    def _random_partition(self, k):
        clusters = [list() for _ in range(k)]
        for p in range(self.ds.shape[0]):
            clusters[randint(0, k)].append(p)
        return list(filter(lambda cl: len(cl) > 0, clusters))

    def _init_single(self):
        partition = self._random_partition(randint(2, self.max_cl))
        clusters = self.mutator.clean_clusters(partition)
        if clusters is None:
            return self._init_single()
        labs = self.stat.unite_by_labels(clusters)
        return labs, self.score(labs)

    def _initialise(self):
        initials = [self._init_single() for _ in range(self.lam)]
        labs, fits = zip(*initials)
        return list(labs), list(fits)

    def _single_step(self, agent, metrics, labels):
        policy, samples = agent.calc_policy(metrics), list()
        arms = np.random.choice(len(policy), size=self.amount, replace=False, p=policy)
        clusters = self.stat.split_by_clusters(labels)
        centroids = self.stat.get_centroids(clusters)
        prototypes = self.stat.get_prototypes(clusters, centroids)
        new_clusters = [self.mutator(arm, labels, clusters, centroids, prototypes) for arm in arms]
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

    def _make_step(self, agent, inputs, labels, fitness):
        used_arms = list()
        for idx, metrics in enumerate(inputs):
            policy, labs, new_metrics, fits, arms = self._single_step(agent, metrics, labels[idx])
            next_gen = None
            if arms is not None:
                agent.backprop(metrics, new_metrics, arms, fits - fitness[idx])
                next_gen = np.argmax(fits)
            if next_gen is None or fits[next_gen] == fitness[idx]:
                labels[idx], fitness[idx] = self._init_single()
                inputs[idx] = self.stat.calc_metrics(labels[idx])
            else:
                labels[idx], fitness[idx], inputs[idx] = labs[next_gen], fits[next_gen], new_metrics[next_gen]
                used_arms.extend(arms)
        return used_arms

    def launch(self, stamps, verbose=True):
        agent = Agent(len(self.mutator.mutations), self.gamma, self.lam)
        start, cur_stamp = time.time(), 0
        results, best_fit, best_labs = list(), float('-inf'), None
        while True:
            labels, fitness = self._initialise()
            best_fit = max(best_fit, *fitness)
            input_metrics = [self.stat.calc_metrics(labs) for labs in labels]
            arms_usage = np.zeros(len(self.mutator.mutations), dtype=int)
            for attempt in range(self.attempts):
                used_arms = self._make_step(agent, input_metrics, labels, fitness)
                for arm in used_arms:
                    arms_usage[arm] += 1
                cur_best = np.argmax(fitness)
                if fitness[cur_best] > best_fit:
                    best_fit, best_labs = fitness[cur_best], labels[cur_best]
                if time.time() - start >= stamps[cur_stamp]:
                    if verbose:
                        print(f'{stamps[cur_stamp]}s: {best_fit}')
                    cur_stamp += 1
                    results.append(best_fit)
                    if cur_stamp == len(stamps):
                        return results, best_labs, best_fit
