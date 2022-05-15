import numpy as np
from numba import njit


class ClusterStat:
    def __init__(self, ds):
        self.ds = ds
        self.global_mst = self._build_mst(np.arange(len(ds), dtype=int))

    @staticmethod
    @njit
    def dist(x, y):
        return np.linalg.norm(x - y)

    def get_centroids(self, clusters):
        return [sum([self.ds[idx] for idx in cl]) / len(cl) if len(cl) != 0 else None for cl in clusters]

    def get_prototypes(self, clusters, centroids=None, idx=False):
        cents = centroids if centroids is not None else self.get_centroids(clusters)
        prototypes = list()
        for centroid, cluster in zip(cents, clusters):
            p_idx = self.dist_to_pivot_(cluster, centroid, np.argmin)
            prototypes.append(cluster[p_idx])
        return prototypes if idx else [self.ds[p_idx] for p_idx in prototypes]

    @staticmethod
    def split_by_clusters(labels):
        lab_max = np.max(labels)
        clusters = [list() for _ in range(lab_max + 1)]
        for idx, sample in enumerate(labels):
            clusters[sample].append(idx)
        return clusters

    def unite_by_labels(self, clusters):
        assert sum(map(len, clusters)) == len(self.ds)
        labs = np.full(len(self.ds), -1, dtype=int)
        for lab, cl in enumerate(clusters):
            for p_idx in cl:
                labs[p_idx] = lab
        return labs

    def dist_to_pivot_(self, cluster, pivot, agg=None):
        dists = [self.dist(self.ds[idx], pivot) for idx in cluster]
        if agg is not None:
            dists = agg.__call__(dists)
        return dists

    def dists_to_pivot(self, clusters, pivots, agg=None):
        return [self.dist_to_pivot_(cluster, centroid, agg) for cluster, centroid in zip(clusters, pivots)]

    def sorted_dist_to_pivot_(self, cluster, pivot, idx=False):
        return sorted([self.dist(self.ds[p_idx], pivot) for p_idx in cluster]) if not idx else \
            sorted([(self.dist(self.ds[p_idx], pivot), p_idx) for p_idx in cluster], key=lambda d: d[0])

    def sorted_dists_to_pivots(self, clusters, pivots, idx=False):
        return [self.sorted_dist_to_pivot_(cluster, pivot, idx) for cluster, pivot in zip(clusters, pivots)]

    def dists_point_to_pivots(self, p_idx, pivots):
        return [self.dist(self.ds[p_idx], centroid) for centroid in pivots]

    def get_self_other(self, label, pivots):
        self_dist = pivots[label]
        pivots[label] = float('inf')
        other_dist = min(*pivots)
        pivots[label] = self_dist
        return self_dist, 1.0 / (1.0 + other_dist)

    def _build_mst(self, cluster):
        n, used, mst = len(cluster), np.zeros(len(cluster), dtype=int), list()
        min_e, sel_e = np.full(n, float('inf')), np.full(n, -1, dtype=int)
        for i in range(n):
            v = -1
            for j in range(n):
                if used[j] == 0 and (v == -1 or min_e[j] < min_e[v]):
                    v = j
            used[v] = 1
            if sel_e[v] != -1:
                mst.append((v, sel_e[v]))
            for to in range(n):
                d = self.dist(self.ds[cluster[v]], self.ds[cluster[to]])
                if d < min_e[to]:
                    min_e[to], sel_e[to] = d, v
        return mst

    def build_mst(self, clusters):
        return [self._build_mst(cluster) for cluster in clusters]

    def calc_metrics(self, labels):
        clusters = self.split_by_clusters(labels)
        prototypes = self.get_prototypes(clusters)
        dist_metrics = list()
        for p_idx, label in enumerate(labels):
            prot_dists = self.dists_point_to_pivots(p_idx, prototypes)
            self_prot, other_prot = self.get_self_other(label, prot_dists)
            dist_metrics.extend([self_prot, other_prot])
        return np.array(dist_metrics)
