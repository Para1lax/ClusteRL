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


def measure_mean_max(logs):
    results, t_len = list(), len(logs['timestamps'])
    for launch in logs['launches']:
        measures = launch['measures'].copy()
        value, m_len = measures[-1], len(measures)
        if m_len < t_len:
            measures.extend([value] * (t_len - m_len))
        results.append(measures)
    mean_results = np.mean(results, axis=0)
    max_result = np.max(np.array(results)[:, -1])
    return mean_results, max_result


def plot_score(evo, rl, ax):
    assert evo['dataset'] == rl['dataset']
    assert evo['measure_fun'] == rl['measure_fun']
    assert evo['measure_fun'] in ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    if evo['measure_fun'] == 'davies_bouldin':
        for launch in evo['launches']:
            for m_idx, m in enumerate(launch['measures']):
                launch['measures'][m_idx] = -m

    evo_mean, evo_max = measure_mean_max(evo)
    rl_mean, rl_max = measure_mean_max(rl)

    ax.set_ylabel(rl['measure_fun'])
    ax.plot(evo['timestamps'], evo_mean, label='(1 + {}): {:.5f}'.format(LAMBDA, evo_max))
    ax.plot(rl['timestamps'], rl_mean, label='RL: {:.5f}'.format(rl_max))
    ax.legend()


ds = pd.read_csv(f'datasets/{DATASET}.csv', header=None)
ds = normalize(np.unique(ds, axis=0), axis=0)
with open(f'results/{DATASET}/evo_runs.dict') as fp:
    evo_logs = list(map(eval, fp.readlines()))
with open(f'results/{DATASET}/rl_runs.dict') as fp:
    rl_logs = list(map(eval, fp.readlines()))

fig, axes = plt.subplots(3, 1)
fig.set_figwidth(7), fig.set_figheight(10)
for m_idx in range(3):
    plot_score(evo_logs[m_idx], rl_logs[m_idx], axes[m_idx])
n, d = ds.shape
fig.suptitle(f'dataset: {DATASET} (objects={n}, dimensions={d})')
fig.supxlabel('time, s')
plt.show()

