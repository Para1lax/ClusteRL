import numpy as np
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, DBSCAN


class ClassicEstimator:
    def __init__(self, ds, measure, inc):
        self.ds, self.measure, self.inc = ds, measure, inc
        self.methods = [self._dbscan, self._kmeans, self._affinity, self._mean_shift]

    def get_result(self, method):
        labels = method.__call__()
        measures = list()
        for labs in labels:
            try:
                measures.append(self.measure(self.ds, labs))
            except ValueError:
                measures.append(float('-inf') if self.inc else float('inf'))
        return max(measures) if self.inc else min(measures)

    def _kmeans(self):
        clusters = range(2, int(np.sqrt(self.ds.shape[0])))
        return [KMeans(k, max_iter=500, tol=1e-8).fit_predict(self.ds) for k in clusters]

    def _affinity(self):
        damps = np.linspace(0.5, 0.9, 10)
        return [AffinityPropagation(damping=damp, max_iter=500).fit_predict(self.ds) for damp in damps]

    def _mean_shift(self):
        return [MeanShift().fit_predict(self.ds) for _ in range(10)]

    def _dbscan(self):
        eps_space = np.linspace(1e-5, 1e-2, 5)
        s_space = np.linspace(2, 10, 5, dtype=int)
        return [DBSCAN(eps=e, min_samples=s).fit_predict(self.ds) for e in eps_space for s in s_space]

    def get_best(self):
        return {method.__name__: self.get_result(method) for method in self.methods}

