import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


matplotlib.use('TkAgg')


def plot_result(results: list, model_name: str, env_name: str, fig_out_path: str):
    results = np.array(results)
    max_t, max_r, max_s, max_sec, max_rt = np.max(results, axis=0).T
    min_t, min_r, min_s, min_sec, min_rt = np.min(results, axis=0).T
    mean_t, mean_r, mean_s, mean_sec, mean_rt = np.mean(results, axis=0).T
    x = np.arange(len(mean_s))

    fig, (axs) = plt.subplots(5, 1, figsize=(8, 16), sharey='none', sharex='all')

    axs[0].plot(max_r, 'y', linewidth=1)
    axs[0].plot(min_r, 'y', linewidth=1)
    axs[0].plot(mean_r, 'y', label=model_name, linewidth=2, linestyle='--')
    axs[0].fill_between(x, min_r, max_r, facecolor='y', edgecolor='y', alpha=0.3)

    axs[1].plot(max_s, 'y', linewidth=1)
    axs[1].plot(min_s, 'y', linewidth=1)
    axs[1].plot(mean_s, 'y', label=model_name, linewidth=2, linestyle='--')
    axs[1].fill_between(x, min_s, max_s, facecolor='y', edgecolor='y', alpha=0.3)

    axs[2].plot(max_t, 'y', linewidth=1)
    axs[2].plot(min_t, 'y', linewidth=1)
    axs[2].plot(mean_t, 'y', label=model_name, linewidth=2, linestyle='--')
    axs[2].fill_between(x, min_t, max_t, facecolor='y', edgecolor='y', alpha=0.3)

    axs[3].plot(max_sec, 'y', linewidth=1)
    axs[3].plot(min_sec, 'y', linewidth=1)
    axs[3].plot(mean_sec, 'y', label=model_name, linewidth=2, linestyle='--')
    axs[3].fill_between(x, min_sec, max_sec, facecolor='y', edgecolor='y', alpha=0.3)

    axs[4].plot(max_rt, 'y', linewidth=1)
    axs[4].plot(min_rt, 'y', linewidth=1)
    axs[4].plot(mean_rt, 'y', label=model_name, linewidth=2, linestyle='--')
    axs[4].fill_between(x, min_rt, max_rt, facecolor='y', edgecolor='y', alpha=0.3)

    axs[0].set_title('Moving Avg Reward (Training)')
    axs[1].set_title('Moving Avg Reward (Evaluation)')
    axs[2].set_title('Cumulated Steps')
    axs[3].set_title('Training Time')
    axs[4].set_title('Wall-clock Time')

    plt.tight_layout()
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')
    plt.savefig(os.path.join(fig_out_path, 'plt_model_{}_env_{}.png'.format(model_name, env_name)), dpi=300, format='png')
    plt.show()


def plt_many_agents(result_path_dict: dict[str, str], fig_out_path: str):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(result_path_dict))]

    fig, (axs) = plt.subplots(5, 1, figsize=(8, 16), sharey='none', sharex='all')

    for idx, (model_name, path) in enumerate(result_path_dict.items()):
        results = np.load(path)
        max_t, max_r, max_s, max_sec, max_rt = np.max(results, axis=0).T
        min_t, min_r, min_s, min_sec, min_rt = np.min(results, axis=0).T
        mean_t, mean_r, mean_s, mean_sec, mean_rt = np.mean(results, axis=0).T
        x = np.arange(len(mean_s))

        axs[0].plot(min_r, color=colors[idx], linewidth=1)
        axs[0].plot(max_r, color=colors[idx], linewidth=1)
        axs[0].plot(mean_r, label=model_name, linewidth=2, color=colors[idx], linestyle='--')
        axs[0].fill_between(x, min_r, max_r, alpha=0.3, facecolor=colors[idx], edgecolor=colors[idx])

        axs[1].plot(min_s, color=colors[idx], linewidth=1)
        axs[1].plot(max_s, color=colors[idx], linewidth=1)
        axs[1].plot(mean_s, label=model_name, linewidth=2, color=colors[idx], linestyle='--')
        axs[1].fill_between(x, min_s, max_s, alpha=0.3, facecolor=colors[idx], edgecolor=colors[idx])

        axs[2].plot(min_t, color=colors[idx], linewidth=1)
        axs[2].plot(max_t, color=colors[idx], linewidth=1)
        axs[2].plot(mean_t, label=model_name, linewidth=2, color=colors[idx], linestyle='--')
        axs[2].fill_between(x, min_t, max_t, alpha=0.3, facecolor=colors[idx], edgecolor=colors[idx])

        axs[3].plot(min_sec, color=colors[idx], linewidth=1)
        axs[3].plot(max_sec, color=colors[idx], linewidth=1)
        axs[3].plot(mean_sec, label=model_name, linewidth=2, color=colors[idx], linestyle='--')
        axs[3].fill_between(x, min_sec, max_sec, alpha=0.3, facecolor=colors[idx], edgecolor=colors[idx])

        axs[4].plot(min_rt, color=colors[idx], linewidth=1)
        axs[4].plot(max_rt, color=colors[idx], linewidth=1)
        axs[4].plot(mean_rt, label=model_name, linewidth=2, color=colors[idx], linestyle='--')
        axs[4].fill_between(x, min_rt, max_rt, alpha=0.3, facecolor=colors[idx], edgecolor=colors[idx])

    axs[0].set_title('Moving Avg Reward (Training)')
    axs[1].set_title('Moving Avg Reward (Evaluation)')
    axs[2].set_title('Cumulated Steps')
    axs[3].set_title('Training Time')
    axs[4].set_title('Wall-clock Time')

    plt.tight_layout()
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')
    plt.savefig(fig_out_path, dpi=300, format='png')
    plt.show()
