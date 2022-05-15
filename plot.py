import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from scipy.stats import wilcoxon


TASK, OBJECT = 'wilcoxon_score', 'all'

COLOURS, LAMBDA = ['b', 'g', 'r', 'c', 'm', 'y', 'k'], r'$\lambda$'
MEASURES = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'gD43']


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


def read_logs(dataset):
    with open(f'results/{dataset}/evo_runs.dict') as fp:
        evo_logs = list(map(eval, fp.readlines()))
        db_evo_log = measure_log(evo_logs, 'davies_bouldin')
        for launch in db_evo_log['launches']:
            for m_idx, m in enumerate(launch['measures']):
                launch['measures'][m_idx] = -m
    with open(f'results/{dataset}/rl_runs.dict') as fp:
        rl_logs = list(map(eval, fp.readlines()))
    return evo_logs, rl_logs


def measure_mean_max(logs):
    results, t_len = list(), len(logs['timestamps'])
    for launch in logs['launches']:
        measures = launch['measures'].copy()
        value, m_len = measures[-1], len(measures)
        if m_len < t_len:
            measures.extend([value] * (t_len - m_len))
        results.append(measures)
    mean_results = np.mean(results, axis=0)
    mean_max = np.mean(np.array(results)[:, -1])
    return mean_results, mean_max


def failed(evo_log):
    return len(evo_log['timestamps']) == 1


def plot_score(evo, rl, ax):
    assert evo['dataset'] == rl['dataset']
    assert evo['measure_fun'] == rl['measure_fun']
    assert evo['timestamps'][-1] == rl['timestamps'][-1] or failed(evo)
    assert evo['measure_fun'] in MEASURES

    evo_means, evo_mean_max = measure_mean_max(evo)
    rl_means, rl_mean_max = measure_mean_max(rl)

    ax.set_ylabel(rl['measure_fun'])
    evo_label = '(1 + {}): {:.5f}'.format(LAMBDA, evo_mean_max)
    if failed(evo):
        ax.tick_params(color='red', labelcolor='red')
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
        ax.plot(rl['timestamps'], list(evo_means) * len(rl['timestamps']), label=evo_label)
    else:
        ax.plot(evo['timestamps'], evo_means, label=evo_label)
    ax.plot(rl['timestamps'], rl_means, label='RL: {:.5f}'.format(rl_mean_max))
    ax.legend()


def single_plot(dataset):
    ds = pd.read_csv(f'datasets/{dataset}.csv', header=None)
    ds = normalize(np.unique(ds, axis=0), axis=0)
    evo_logs, rl_logs = read_logs(dataset)

    fig, axes = plt.subplots(2, 2)
    axes = np.ravel(axes)
    fig.set_figwidth(7), fig.set_figheight(10)
    for m_idx in range(4):
        plot_score(evo_logs[m_idx], rl_logs[m_idx], axes[m_idx])
    n, d = ds.shape
    fig.suptitle(f'dataset: {dataset} (objects={n}, dimensions={d})')
    fig.supxlabel('time, s')
    plt.show()


def measure_log(logs, measure_fun):
    for log in logs:
        if log['measure_fun'] == measure_fun:
            return log
    raise ValueError(f"No {measure_fun} logs for {logs['dataset']}")


def wilcoxon_score(measure_fun):
    if measure_fun == 'all':
        return [wilcoxon_score(measure) for measure in MEASURES]
    evo_scores, rl_scores = list(), list()
    for dataset_dir in os.listdir('results'):
        files = set(os.listdir(f'results/{dataset_dir}'))
        if {'evo_runs.dict', 'rl_runs.dict', 'plots.png'}.issubset(files):
            evo_logs, rl_logs = read_logs(dataset_dir)
            try:
                evo, rl = measure_log(evo_logs, measure_fun), measure_log(rl_logs, measure_fun)
                evo_means, evo_mean_max = measure_mean_max(evo)
                rl_means, rl_mean_max = measure_mean_max(rl)
                evo_scores.append(evo_mean_max), rl_scores.append(rl_mean_max)
            except ValueError as e:
                print(e.__str__())
    print(f'datasets: {len(rl_scores)}, measure: {measure_fun}')
    print(wilcoxon(rl_scores, evo_scores, alternative='less'))


if __name__ == '__main__':
    globals()[TASK].__call__(OBJECT)
