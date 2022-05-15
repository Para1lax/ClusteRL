import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from executor import Executor
from mutations import Mutator
from stats import ClusterStat


DS_NAME, MEASURE_FUN = '100-plants-margin', 'davies_bouldin'
BUDGET, GRID, LAUNCHES = 280, 40, 3


def dump(ds, timestamps, launches):
    dumper = {
        "dataset": DS_NAME,
        "shape": ds.shape,
        "measure_fun": MEASURE_FUN,
        "timestamps": list(timestamps),
        "launches": list()
    }
    for measures, partition in launches:
        dumper['launches'].append({"measures": list(measures), "best_partition": list(partition)})
    return dumper


ds = pd.read_csv(f'datasets/{DS_NAME}.csv', header=None)
ds = normalize(np.unique(ds, axis=0), axis=0)

stat, stamps = ClusterStat(ds), np.linspace(BUDGET / GRID, BUDGET, GRID)
mutator = Mutator(ds, stat, use_best=False)
executor = Executor(ds, stat, mutator, measure=MEASURE_FUN, attempts=64, lam=4, amount=4)

results = list()
for launch in range(LAUNCHES):
    vals, labs, fit = executor.launch(stamps, verbose=True)
    results.append((vals, labs))
print(dump(ds, stamps, results).__str__())
