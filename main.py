import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from mutations import Mutator
from stats import ClusterStat
from executor import Executor


ds = pd.read_csv('datasets/vehicle.csv', header=None)
ds = normalize(np.unique(ds, axis=0), axis=0)
BUDGET, GRID, LAUNCHES = 500, 50, 5
stat, stamps = ClusterStat(ds), np.linspace(BUDGET / GRID, BUDGET, GRID)
mutator = Mutator(ds, stat, use_best=False)
executor = Executor(ds, stat, mutator, measure='sil', attempts=64, lam=8, gamma=0.9)

results = np.array([executor.launch(stamps) for _ in range(LAUNCHES)])
pd.DataFrame(results, columns=stamps).to_csv('results/vehicle_sil.csv')
