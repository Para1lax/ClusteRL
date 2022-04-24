import numpy as np
from numba import njit


class ClusterStat:
    def __init__(self, ds):
        self.ds = ds

    @staticmethod
    @njit
    def dist(x, y):
        return np.linalg.norm(x - y)

    def get_centroids_by_labels(self, labels):
        lab_max = np.max(labels)
        centroids = [np.zeros(self.ds.shape[1]) for _ in range(lab_max + 1)]
        counts = np.zeros(lab_max + 1, int)
        for p_idx, lab in enumerate(labels):
            cur_sum = centroids[lab] * counts[lab] + self.ds[p_idx]
            counts[lab] += 1
            centroids[lab] = cur_sum / counts[lab]
        return centroids

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

    def dists_to_clusters(self, centroids, major='cluster'):
        if major == 'cluster':
            return [[self.dist(self.ds[idx], centroid) for idx in range(len(self.ds))] for centroid in centroids]
        elif major == 'point':
            return [[self.dist(self.ds[idx], centroid) for centroid in centroids] for idx in range(len(self.ds))]
        else:
            raise ValueError(f"Expected 'cluster' or 'point', got {major}")

    def get_self_other(self, label, pivots):
        self_dist = pivots[label]
        pivots[label] = float('inf')
        other_dist = min(*pivots)
        pivots[label] = self_dist
        return self_dist, 1.0 / (1.0 + other_dist)

    def calc_metrics(self, labels):
        # laplacian = np.zeros(shape=(len(self.ds), len(self.ds)))
        # for p_idx, label in enumerate(labels):
        #     sum_affinity = 0
        #     for p_other, other_label in enumerate(labels):
        #         if label == other_label and p_idx != p_other:
        #             aff = 1.0 / (1e-8 + self.dist(self.ds[p_idx], self.ds[p_other]))
        #             laplacian[p_idx, p_other] = -aff
        #             sum_affinity += aff
        #     laplacian[p_idx, p_idx] = sum_affinity
        # eigen, _ = np.linalg.eigh(laplacian)
        clusters = self.split_by_clusters(labels)
        centroids = self.get_centroids(clusters)
        prototypes = self.get_prototypes(clusters, centroids=centroids)
        dist_metrics = list()
        for p_idx, label in enumerate(labels):
            cent_dists = self.dists_point_to_pivots(p_idx, centroids)
            self_cent, other_cent = self.get_self_other(label, cent_dists)
            prot_dists = self.dists_point_to_pivots(p_idx, prototypes)
            self_prot, other_prot = self.get_self_other(label, prot_dists)
            dist_metrics.extend([self_cent, self_prot, other_cent, other_prot])
        return np.array(dist_metrics)
        # return np.concatenate([eigen, np.array(dist_metrics)])
