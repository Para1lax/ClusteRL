import copy
import numpy as np

from numpy.random import choice
from itertools import combinations
from scipy.stats import tmean


class Mutator:
    def __init__(self, ds, stat, few=0.02, lot=0.09, sigma=0.02, min_size=2, eps=1.0, use_best=True):
        self.ds, self.stat, self.dist, self.best = ds, stat, stat.dist, use_best
        self.eps, self.min_size = eps, min_size
        self.few, self.lot, self.sigma = few, lot, sigma
        self.mutations = [
            self.far_split, self.half_split, self.triple_split,
            self.new_prototype_by_centroid, self.new_prototype_by_prototype,
            self.delete_by_centroid, self.delete_by_prototype,
            self.merge_by_centroid, self.merge_by_prototype, self.merge_by_global_mst,
            self.move_few_from_cluster_by_centroid, self.move_lot_from_cluster_by_centroid,
            #self.move_few_from_cluster_by_prototype, self.move_lot_from_cluster_by_prototype,
            # self.move_single_by_centroid,
            self.move_few_rate_by_centroid, self.move_lot_rate_by_centroid,
            #self.move_single_by_prototype, self.move_lot_rate_by_prototype, self.move_lot_rate_by_prototype,
            # self.expand_single_by_centroid,
            self.expand_few_by_centroid, self.expand_lot_by_centroid,
            #self.expand_single_by_prototype, self.expand_few_by_prototype, self.expand_lot_by_prototype
        ]

    def _norm(self, rates):
        r_max = np.max(rates)
        soft_rates = np.array([1.5 ** (r - r_max) for r in rates])
        return soft_rates / soft_rates.sum()

    def _idx(self, seq, p):
        return max(1, min(int(len(seq) * (1.0 - p)), len(seq) - 2))

    def _sample_p(self, mean):
        return np.random.uniform(mean - self.sigma, mean + self.sigma)

    # def _clean_delete(self, clusters, deleter_idx, centroids):
    #     deleter = clusters[deleter_idx]
    #     del clusters[deleter_idx], centroids[deleter_idx]
    #     for p_idx in deleter:
    #         p_dists = self.stat.dists_point_to_pivots(p_idx, centroids)
    #         new_label = np.argmin(p_dists)
    #         clusters[new_label].append(p_idx)
    #     return clusters

    def clean_clusters(self, clusters):
        if clusters is None:
            return None
        clusters = list(filter(lambda cluster: len(cluster) != 0, clusters))
        # centroids = self.stat.get_centroids(clusters)
        # removable_clusters = list()
        # for idx, cl in enumerate(clusters):
        #     if len(cl) < self.min_size:
        #         removable_clusters.append(idx)
        # if len(removable_clusters) == 0:
        #     return clusters
        # temp_cluster = list()
        # for idx in reversed(removable_clusters):
        #     temp_cluster.extend(clusters[idx])
        #     del clusters[idx], centroids[idx]
        # clusters.append(temp_cluster), centroids.append(None)
        # self._clean_delete(clusters, len(clusters) - 1, centroids)
        return None if len(clusters) < 2 else clusters

    def _get_farthest(self, spliterator, centroid):
        far_idx = spliterator[self.stat.dist_to_pivot_(spliterator, centroid, np.argmax)]
        far_far = np.argmax([self.dist(self.ds[idx], self.ds[far_idx]) for idx in spliterator])
        return self.ds[far_idx], self.ds[spliterator[far_far]]

    def half_split(self, **kwargs):
        clusters, centroids = kwargs['clusters'], kwargs['centroids']
        dists = self.stat.dists_to_pivot(clusters, centroids, tmean)
        split_idx = np.argmax(dists) if self.best else choice(len(dists), p=self._norm(dists))
        p1, p2 = self._get_farthest(clusters[split_idx], centroids[split_idx])

        new_clusters, moved = copy.deepcopy(clusters), 0
        new_clusters.append(list())
        for idx, p in enumerate(clusters[split_idx]):
            if self.dist(self.ds[p], p1) > self.dist(self.ds[p], p2):
                del new_clusters[split_idx][idx - moved]
                moved += 1
                new_clusters[-1].append(p)
        return new_clusters

    def triple_split(self, **kwargs):
        clusters, centroids = kwargs['clusters'], kwargs['centroids']
        dists = self.stat.dists_to_pivot(clusters, centroids, np.mean)
        split_idx = np.argmax(dists) if self.best else choice(len(dists), p=self._norm(dists))
        p1, p2 = self._get_farthest(clusters[split_idx], centroids[split_idx])

        new_clusters, moved = copy.deepcopy(clusters), 0
        new_clusters.extend([list(), list()])
        for idx, p in enumerate(clusters[split_idx]):
            d1, d2 = self.dist(self.ds[p], p1), self.dist(self.ds[p], p2)
            dc = self.dist(self.ds[p], centroids[split_idx])
            new_lab = np.argmin([dc, d1, d2])
            if new_lab != 0:
                del new_clusters[split_idx][idx - moved]
                moved += 1
                new_clusters[-new_lab].append(p)
        return new_clusters

    def far_split(self, **kwargs):
        clusters, centroids = kwargs['clusters'], kwargs['centroids']
        dists = self.stat.dists_to_pivot(clusters, centroids, np.mean)
        split_idx = np.argmax(dists) if self.best else choice(len(dists), p=self._norm(dists))

        spliterator, centroid = clusters[split_idx], centroids[split_idx]
        sp_idx = self.stat.dist_to_pivot_(spliterator, centroid, np.argmax)
        the_farthest = self.ds[spliterator[sp_idx]]

        new_clusters, moved = copy.deepcopy(clusters), 0
        new_clusters.append(list())
        for idx, p in enumerate(spliterator):
            if self.dist(self.ds[p], centroid) > self.dist(self.ds[p], the_farthest):
                del new_clusters[split_idx][idx - moved]
                moved += 1
                new_clusters[-1].append(p)
        return new_clusters

    def _merge_rate(self, centroids, indices):
        return sum([self.dist(centroids[x], centroids[y]) for x, y in combinations(indices, 2)])

    # def _merge(self, clusters, k_merge):
    #     if len(clusters) <= k_merge:
    #         return None
    #     centroids, comb = self.stat.get_centroids(clusters), combinations(range(len(clusters)), k_merge)
    #     dists = [(self._merge_rate(centroids, indices), indices) for indices in comb]
    #     rates = [1.0 / (self.eps + d[0]) for d in dists]
    #     m_idx = np.argmax(rates) if self.best else choice(len(rates), p=self._norm(rates))
    #     mergers = list(sorted(dists[m_idx][1], reverse=True))
    #     new_clusters, acc = copy.deepcopy(clusters), mergers[-1]
    #     for merger in mergers[:-1]:
    #         del new_clusters[merger]
    #         new_clusters[acc].extend(clusters[merger])
    #     return new_clusters

    def merge_by_global_mst(self, **kwargs):
        clusters, labels = kwargs['clusters'], kwargs['labels']
        bridges = list(filter(lambda mst_edge: labels[mst_edge[0]] != labels[mst_edge[1]], self.stat.global_mst))
        bridge_rates = [1.0 / (1.0 + self.dist(self.ds[x], self.ds[y])) for x, y in bridges]
        bridge_idx = np.argmax(bridge_rates) if self.best else choice(len(bridges), p=self._norm(bridge_rates))
        x, y = bridges[bridge_idx]
        m1, m2 = labels[x], labels[y]

        new_clusters = copy.deepcopy(clusters)
        del new_clusters[max(m1, m2)]
        new_clusters[min(m1, m2)].extend(clusters[max(m1, m2)])
        return new_clusters

    def merge_by_prototype(self, **kwargs):
        return self._merge_by_pivots(kwargs['clusters'], kwargs['prototypes'])

    def merge_by_centroid(self, **kwargs):
        return self._merge_by_pivots(kwargs['clusters'], kwargs['centroids'])

    def _merge_by_pivots(self, clusters, pivots):
        if len(clusters) <= 2:
            return None
        comb = combinations(range(len(pivots)), 2)
        dists = [(self.dist(pivots[p1], pivots[p2]), (p1, p2)) for p1, p2 in comb]
        rates = [1.0 / (self.eps + d[0]) for d in dists]
        m_idx = np.argmax(rates) if self.best else choice(len(rates), p=self._norm(rates))
        m1, m2 = dists[m_idx][1]
        new_clusters = copy.deepcopy(clusters)
        del new_clusters[m2]
        new_clusters[m1].extend(clusters[m2])
        return new_clusters

    def _delete_by_pivot(self, clusters, pivots):
        if len(clusters) <= 2:
            return None
        rates, p = list(), (self.few + self.lot) / 2
        dists = self.stat.sorted_dists_to_pivots(clusters, pivots)
        for sort_dists in dists:
            rates.append(0.0 if len(sort_dists) < 2 else tmean(sort_dists[:self._idx(sort_dists, p)]))
        del_idx = np.argmax(rates) if self.best else choice(len(rates), p=self._norm(rates))

        new_clusters, removers, new_pivots = copy.deepcopy(clusters), list(), copy.deepcopy(pivots)
        del new_clusters[del_idx], new_pivots[del_idx]
        for p_idx in clusters[del_idx]:
            p_dists = self.stat.dists_point_to_pivots(p_idx, new_pivots)
            new_label = np.argmin(p_dists)
            new_clusters[new_label].append(p_idx)
        return new_clusters

    def delete_by_centroid(self, **kwargs):
        return self._delete_by_pivot(kwargs['clusters'], kwargs['centroids'])

    def delete_by_prototype(self, **kwargs):
        return self._delete_by_pivot(kwargs['clusters'], kwargs['prototypes'])

    # def _delete(self, clusters, k_delete):
    #     if len(clusters) <= k_delete + 1:
    #         return None
    #     centroids, rates = self.stat.get_centroids(clusters), list()
    #     p = (self.few + self.lot) / 2
    #     dists = self.stat.sorted_dists_to_pivots(clusters, centroids)
    #     for sort_dists in dists:
    #         p_idx = self._idx(sort_dists, p)
    #         rates.append(tmean(sort_dists[:p_idx]))
    #     if self.best:
    #         temp = sorted(zip(rates, range(len(rates))), key=lambda r: r[0], reverse=True)
    #         _, deletes = zip(*temp[:k_delete])
    #         deletes = list(deletes)
    #     else:
    #         deletes = choice(len(rates), k_delete, replace=False, p=self._norm(rates)).tolist()
    #
    #     deletes.sort(reverse=True)
    #     new_clusters, removers = copy.deepcopy(clusters), list()
    #     for del_idx in deletes:
    #         removers.extend(clusters[del_idx])
    #         del new_clusters[del_idx], centroids[del_idx]
    #     for p_idx in removers:
    #         p_dists = self.stat.dists_point_to_pivots(p_idx, centroids)
    #         new_label = np.argmin(p_dists)
    #         new_clusters[new_label].append(p_idx)
    #     return new_clusters

    def _move_by_pivot(self, clusters, p, pivots):
        dists_indices, rates = self.stat.sorted_dists_to_pivots(clusters, pivots, idx=True), list()
        for sort_dists in dists_indices:
            dists, _ = zip(*sort_dists)
            rates.append(0.0 if len(sort_dists) < 2 else tmean(list(dists)[self._idx(sort_dists, p):]))
        m_idx = np.argmax(rates) if self.best else choice(len(rates), p=self._norm(rates))

        _, movers = zip(*dists_indices[m_idx])
        new_clusters, new_pivots = copy.deepcopy(clusters), copy.deepcopy(pivots)
        if len(movers) < 2:
            p_dists = self.stat.dists_point_to_pivots(movers[0], pivots)
            del p_dists[m_idx], new_clusters[m_idx]
            new_label = np.argmin(p_dists)
            new_clusters[new_label].append(movers[0])
            return new_clusters
        movers, q_idx = list(movers), self._idx(movers, p)
        del new_clusters[m_idx], new_pivots[m_idx]
        for p_idx in movers[q_idx:]:
            p_dists = self.stat.dists_point_to_pivots(p_idx, new_pivots)
            # rates = [1.0 / (self.eps + d) for d in p_dists]
            new_label = np.argmin(p_dists) # choice(len(rates), p=self._norm(rates))
            new_clusters[new_label].append(p_idx)
        new_clusters.append(list())
        for p_idx in movers[:q_idx]:
            new_clusters[-1].append(p_idx)
        return new_clusters

    # def _move(self, clusters, p):
    #     centroids, rates = self.stat.get_centroids(clusters), list()
    #     dists_indices = self.stat.sorted_dists_to_pivots(clusters, centroids, idx=True)
    #     for sort_dists in dists_indices:
    #         p_idx = self._idx(sort_dists, p)
    #         dists, _ = zip(*sort_dists)
    #         rates.append(tmean(list(dists)[p_idx:]))
    #     m_idx = np.argmax(rates) if self.best else choice(len(rates), p=self._norm(rates))
    #
    #     _, movers = zip(*dists_indices[m_idx])
    #     movers, q_idx = list(movers), self._idx(movers, p)
    #     new_clusters = copy.deepcopy(clusters)
    #     del new_clusters[m_idx], centroids[m_idx]
    #     for p_idx in movers[q_idx:]:
    #         p_dists = self.stat.dists_point_to_pivots(p_idx, centroids)
    #         # rates = [1.0 / (self.eps + d) for d in p_dists]
    #         new_label = np.argmin(p_dists)
    #         new_clusters[new_label].append(p_idx)
    #     new_clusters.append(list())
    #     for p_idx in movers[:q_idx]:
    #         new_clusters[-1].append(p_idx)
    #     return new_clusters

    def move_few_from_cluster_by_centroid(self, **kwargs):
        return self._move_by_pivot(kwargs['clusters'], self._sample_p(self.few), kwargs['centroids'])

    def move_lot_from_cluster_by_centroid(self, **kwargs):
        return self._move_by_pivot(kwargs['clusters'], self._sample_p(self.lot), kwargs['centroids'])

    def move_few_from_cluster_by_prototype(self, **kwargs):
        return self._move_by_pivot(kwargs['clusters'], self._sample_p(self.few), kwargs['prototypes'])

    def move_lot_from_cluster_by_prototype(self, **kwargs):
        return self._move_by_pivot(kwargs['clusters'], self._sample_p(self.lot), kwargs['prototypes'])

    def _p_rates(self, labels, pivots):
        rates = list()
        for p_idx, lab in enumerate(labels):
            dists = self.stat.dists_point_to_pivots(p_idx, pivots)
            self_dist = dists[lab]
            dists[lab] = float('inf')
            other_dist = min(*dists)
            rates.append(self_dist / (1.0 + other_dist))
        return rates

    def _move_by_rate(self, labels, p, pivots):
        rates = list(zip(range(len(labels)), self._p_rates(labels, pivots)))
        move_size = max(1, int(len(labels) * p))
        rates.sort(key=lambda r: r[1], reverse=True)
        movers, _ = zip(*rates)
        new_labels = copy.deepcopy(labels)
        for p_idx in list(movers)[:move_size]:
            p_dists = self.stat.dists_point_to_pivots(p_idx, pivots)
            rates = [1.0 / (1.0 + d) for d in p_dists]
            new_label = choice(len(rates), p=self._norm(rates))
            new_labels[p_idx] = new_label
        return self.stat.split_by_clusters(new_labels)

    def move_few_rate_by_centroid(self, **kwargs):
        return self._move_by_rate(kwargs['labels'], self._sample_p(self.few), kwargs['centroids'])

    def move_lot_rate_by_centroid(self, **kwargs):
        return self._move_by_rate(kwargs['labels'], self._sample_p(self.lot), kwargs['centroids'])

    def move_few_rate_by_prototype(self, **kwargs):
        return self._move_by_rate(kwargs['labels'], self._sample_p(self.few), kwargs['prototypes'])

    def move_lot_rate_by_prototype(self, **kwargs):
        return self._move_by_rate(kwargs['labels'], self._sample_p(self.lot), kwargs['prototypes'])

    def _new_prototype_by_pivot(self, clusters, labels, pivots):
        p_best = np.argmax(self._p_rates(labels, pivots))
        new_clusters = [list() for _ in range(len(clusters) + 1)]
        for p_idx, lab in enumerate(labels):
            p, c = self.ds[p_idx], pivots[lab]
            if self.dist(p, c) < self.dist(p, self.ds[p_best]):
                new_clusters[lab].append(p_idx)
            else:
                new_clusters[-1].append(p_idx)
        return new_clusters

    def new_prototype_by_centroid(self, **kwargs):
        return self._new_prototype_by_pivot(kwargs['clusters'], kwargs['labels'], kwargs['centroids'])

    def new_prototype_by_prototype(self, **kwargs):
        return self._new_prototype_by_pivot(kwargs['clusters'], kwargs['labels'], kwargs['prototypes'])

    def _expand_by_pivots(self, clusters, labels, p, pivots):
        rates, dists = list(), self.stat.sorted_dists_to_pivots(clusters, pivots, idx=True)
        for sort_dists in dists:
            rate = 0.0 if len(sort_dists) < 2 else tmean(sort_dists[:self._idx(sort_dists, p)])
            rates.append(1.0 / (rate + self.eps))
        e_idx = np.argmax(rates) if self.best else choice(len(rates), p=self._norm(rates))
        others = list(filter(lambda idx: labels[idx] != e_idx, range(len(self.ds))))
        if len(others) < 2:
            return None
        new_clusters, c = [list() for _ in range(len(clusters))], pivots[e_idx]
        others_dists = [self.dist(self.ds[p_idx], c) for p_idx in others]
        q_idx = self._idx(others_dists, p)
        quantile_dist = (sum(others_dists[:q_idx]) * (1.0 - p) ** 2) / np.sqrt(self.ds.shape[1])
        p_others, acc_dist = sorted(zip(others_dists, others), key=lambda d: d[0]), 0.0
        for d, p_idx in p_others:
            acc_dist += d
            if acc_dist <= quantile_dist:
                new_clusters[e_idx].append(p_idx)
            else:
                new_clusters[labels[p_idx]].append(p_idx)
        new_clusters[e_idx].extend(clusters[e_idx])
        return new_clusters

    def expand_few_by_centroid(self, **kwargs):
        return self._expand_by_pivots(
            kwargs['clusters'], kwargs['labels'], self._sample_p(self.few), kwargs['centroids']
        )

    def expand_lot_by_centroid(self, **kwargs):
        return self._expand_by_pivots(
            kwargs['clusters'], kwargs['labels'], self._sample_p(self.lot), kwargs['centroids']
        )

    def expand_few_by_prototype(self, **kwargs):
        return self._expand_by_pivots(
            kwargs['clusters'], kwargs['labels'], self._sample_p(self.few), kwargs['prototypes']
        )

    def expand_lot_by_prototype(self, **kwargs):
        return self._expand_by_pivots(
            kwargs['clusters'], kwargs['labels'], self._sample_p(self.lot), kwargs['prototypes']
        )

    def __call__(self, arm, labels, clusters, centroids, prototypes):
        new_gen = self.mutations[arm].__call__(
            labels=labels, clusters=clusters, centroids=centroids, prototypes=prototypes
        )
        return self.clean_clusters(new_gen)
