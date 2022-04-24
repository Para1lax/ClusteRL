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
        'sil': (silhouette_score, True),
        'db': (davies_bouldin_score, False),
        'ch': (calinski_harabasz_score, True)
    }

    def __init__(self, ds, stat, mutator, measure, *, attempts=70, init='random', gamma=0.95, lam=8, max_cl=25):
        self.ds, self.stat, self.mutator, self.max_cl = ds, stat, mutator, max_cl
        self.init_partition = self.kmeans_partition if init == 'k-means' else self.random_partition
        self.measure, self.inc = self.measure_map[measure]
        self.attempts, self.gamma, self.lam = attempts, gamma, lam

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

    def make_step(self, agent, input_metrics, labels, fitness, arms_usage):
        arms, curs, outs, actions, rewards = [], [], [], [], []
        for metric in input_metrics:
            arm = agent.sample_action(metric)
            arms_usage[arm] += 1
            arms.append(arm)
        arms = [agent.sample_action(metric) for metric in input_metrics]
        new_clusters = [self.mutator(labs, arm) for labs, arm in zip(labels, arms)]
        for idx, partition in enumerate(new_clusters):
            if partition is None:
                labels[idx], fitness[idx] = self.init_single()
                input_metrics[idx] = self.stat.calc_metrics(labels[idx])
            else:
                labels[idx] = self.stat.unite_by_labels(partition)
                new_fit, new_metrics = self.score(labels[idx]), self.stat.calc_metrics(labels[idx])
                curs.append(input_metrics[idx]), outs.append(new_metrics)
                actions.append(arms[idx]), rewards.append(new_fit - fitness[idx])
                fitness[idx], input_metrics[idx] = new_fit, new_metrics
        if len(rewards) > 1:
            best_idx = np.argmax(rewards)
            agent.backprop(np.array(curs), np.array(outs), np.array(actions), np.array(rewards))
            return curs[best_idx], outs[best_idx], actions[best_idx], rewards[best_idx]
        return None

    def launch(self, stamps):
        agent = Agent(len(self.mutator.mutations), self.gamma, self.lam)
        start, cur_stamp, epoch = time.time(), 0, 0
        results, the_best = list(), float('-inf')
        while True:
            labels, fitness = self.initialise()
            input_metrics = [self.stat.calc_metrics(labs) for labs in labels]
            arms_usage, replays = np.zeros(len(self.mutator.mutations), dtype=int), list()
            epoch_best, epoch = max(*fitness), epoch + 1
            for attempt in range(self.attempts):
                record = self.make_step(agent, input_metrics, labels, fitness, arms_usage)
                if record is not None:
                    replays.append(record)
                the_best, epoch_best = max(the_best, *fitness), max(epoch_best, *fitness)
                if time.time() - start >= stamps[cur_stamp]:
                    print(f'{int(stamps[cur_stamp])}s: {the_best}')
                    cur_stamp += 1
                    results.append(the_best)
                    if cur_stamp == len(stamps):
                        return results
                # print(', '.join('{:.3f}'.format(f) for f in fitness))
            for replay in np.array_split(np.asarray(replays, dtype=object), self.lam):
                inputs, outs, actions, rewards = zip(*replay)
                agent.backprop(np.array(inputs), np.array(outs), np.array(actions), np.array(rewards))
            print(f"=== BEST FOR EPOCH #{epoch} is {epoch_best} ===")
            print(arms_usage)
