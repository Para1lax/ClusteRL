import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def mean_max(stamps):
    return np.nanmean(stamps.values, axis=0), np.nanmax(stamps.values)


DATASET, MEASURE = 'yeast', 'ch'

n, d = pd.read_csv(f'datasets/{DATASET}.csv').shape

evo_rl = pd.read_csv(f'results/{DATASET}/rl_evo_{MEASURE}.csv')
evo_rl_mean, evo_rl_max = mean_max(evo_rl)

with open(f'results/{DATASET}/mab_{MEASURE}.txt') as f:
    times, measures = eval(f.readline()), eval(f.readline())
mab_times, mab_vals = list(), list()
for attempt, values in enumerate(measures):
    idx = np.argmax(values)
    mab_times.append(times[attempt][idx])
    mab_vals.append(values[idx])
not_null = list(filter(lambda tm: tm[1] != 0.0, zip(times, measures)))
times, measures = zip(*not_null)

timestamps = np.array(evo_rl.columns, dtype=float)

plt.plot(timestamps, evo_rl_mean, label='RL Evolutionary: {:.5f}'.format(evo_rl_max))
plt.plot(mab_times, mab_vals, 'go', label='RL Heuristic: {:.5f}'.format(np.max(mab_vals)))
plt.title(f'{DATASET} (N={n}, dim={d}), {MEASURE}')
plt.xlabel('time, s')
plt.ylabel('measure')
plt.legend()
plt.show()
