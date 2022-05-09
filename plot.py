import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

DATASET = 'pollen'
COLOURS, LAMBDA = ['b', 'g', 'r', 'c', 'm', 'y', 'k'], r'$\lambda$'


def pca_collapse(dataset, to_dim):
    values, vectors = np.linalg.eig(dataset.transpose().dot(dataset))
    marked_values = [(values[x], x) for x in range(dataset.shape[1])]
    marked_values.sort(key=lambda x: abs(x[0]), reverse=True)
    collapsed_transform = np.array([vectors[:, marked_values[x][1]] for x in range(to_dim)]).transpose()
    info = np.sum([marked_values[x][0] for x in range(to_dim)]) / np.sum(values)
    return dataset.dot(collapsed_transform), info


def plot_pca(dataset, labels):
    ds_map, info = pca_collapse(dataset, 2)
    colours = [COLOURS[lab] for lab in labels]
    plt.scatter(ds_map[:, 0], ds_map[:, 1], c=colours)
    plt.show()


def plot_umap(dataset, labels):
    from umap import UMAP
    import umap.plot
    mapper = UMAP(n_neighbors=int(np.sqrt(len(dataset))), min_dist=1e-5).fit(dataset)
    p = umap.plot.points(mapper, labels)
    p.plot()
    plt.show()


def mean_max(stamps):
    return np.nanmean(stamps.values, axis=0), np.nanmax(stamps.values)


def find_measure_argmax(logs):
    best_launch, timestamp, measure = None, float('inf'), float('-inf')
    for launch_idx, launch in enumerate(logs['launches']):
        attempt_max = launch['measures'][-1]
        if attempt_max >= measure:
            measure, t = attempt_max, 0
            while launch['measures'][t] != attempt_max:
                t += 1
            if logs['timestamps'][t] < timestamp:
                timestamp, best_launch = logs['timestamps'][t], launch
    t_len, m_len = len(logs['timestamps']), len(best_launch['measures'])
    if m_len < t_len:
        best_launch['measures'].extend([measure] * (t_len - m_len))
    return best_launch, timestamp, measure


def plot_score(evo, rl, ax):
    assert evo['dataset'] == rl['dataset']
    assert evo['measure_fun'] == rl['measure_fun']
    assert evo['measure_fun'] in ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    if evo['measure_fun'] == 'davies_bouldin':
        for launch in evo['launches']:
            for m_idx, m in enumerate(launch['measures']):
                launch['measures'][m_idx] = -m
    evo_launch, evo_time, evo_measure = find_measure_argmax(evo)
    rl_launch, rl_time, rl_measure = find_measure_argmax(rl)

    evo_cluster_sizes = np.unique(evo_launch['best_partition'], return_counts=True)[1]
    singe_obj_clusters = filter(lambda k_size: k_size == 1, evo_cluster_sizes)
    if next(singe_obj_clusters, None) is not None:
        ax.tick_params(color='red', labelcolor='red')
        for spine in ax.spines.values():
            spine.set_edgecolor('red')

    ax.set_ylabel(rl['measure_fun'])
    ax.plot(evo['timestamps'], evo_launch['measures'], label='(1 + {}): {:.5f}'.format(LAMBDA, evo_measure))
    ax.plot(rl['timestamps'], rl_launch['measures'], label='RL: {:.5f}'.format(rl_measure))
    ax.legend()


ds = pd.read_csv(f'datasets/{DATASET}.csv', header=None)
ds = normalize(np.unique(ds, axis=0), axis=0)
with open(f'results/{DATASET}/evo_results.dict') as fp:
    evo_logs = list(map(eval, fp.readlines()))
with open(f'results/{DATASET}/rl_results.dict') as fp:
    rl_logs = list(map(eval, fp.readlines()))

fig, axes = plt.subplots(3, 1)
fig.set_figwidth(7), fig.set_figheight(10)
for m_idx in range(3):
    plot_score(evo_logs[m_idx], rl_logs[m_idx], axes[m_idx])
n, d = ds.shape
fig.suptitle(f'dataset: {DATASET} (objects={n}, dimensions={d})')
fig.supxlabel('time, s')
plt.show()

# with open(f'results/{DATASET}/mab_{MEASURE}.txt') as f:
#     times, measures = eval(f.readline()), eval(f.readline())
# mab_times, mab_vals = list(), list()
# for attempt, values in enumerate(measures):
#     idx = np.argmax(values)
#     mab_times.append(times[attempt][idx])
#     mab_vals.append(values[idx])
# not_null = list(filter(lambda tm: tm[1] != 0.0, zip(times, measures)))
# times, measures = zip(*not_null)
#
# timestamps = np.array(evo_rl.columns, dtype=float)
#
# plt.plot(timestamps, evo_rl_mean, label='RL Evolutionary: {:.5f}'.format(evo_rl_max))
# plt.plot(mab_times, mab_vals, 'go', label='RL Heuristic: {:.5f}'.format(np.max(mab_vals)))
# plt.title(f'{DATASET} (N={n}, dim={d}), {MEASURE}')
# plt.legend()
# plt.show()
