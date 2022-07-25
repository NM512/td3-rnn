import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse

COLOR_LIST = ["r", "g", "b", "c", "m", "y", "k"]

def load_files_recursively(path, label):
    concated_data = []
    for curDir, dirs, files in os.walk(path):
        for file in files:
            if file == 'progress.txt':
                data = pd.read_table(os.path.join(curDir, file))[label]
                concated_data.append(np.array(data)[np.newaxis,:])
    concated_data = np.concatenate(concated_data, axis=0)
    return concated_data

def plot_learning_curve(data, color, path, legend_name=None):
    if len(data) > 1:
        score_mean = np.mean(data, axis=0)
        score_std = np.std(data, axis=0)
        plt.fill_between(
            range(len(score_mean)),
            score_mean - score_std,
            score_mean + score_std,
            alpha=0.1,
            color=color)
        if legend_name is None:
            plt.plot(range(len(score_mean)), score_mean, color=color, label=path.split('/')[-1])
        else:
            plt.plot(range(len(score_mean)), score_mean, color=color, label=legend_name)
    else:
        if legend_name is None:
            plt.plot(range(len(data[0])), data[0], color=color, label=path.split('/')[-1])
        else:
            plt.plot(range(len(data[0])), data[0], color=color, label=legend_name)
    return plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='an example program')
    parser.add_argument('--plist', required=True, nargs="*", type=str)
    parser.add_argument('--label', type=str, default='AverageTestEpRet')
    parser.add_argument('--legend_name_list', type=str, nargs="*", default=None)

    args = parser.parse_args()

    title = "Learning Curves"
    for i, path in enumerate(args.plist):
        data = load_files_recursively(
            path=path, 
            label=args.label
            )
        if args.legend_name_list is not None:
            plot_learning_curve(data, COLOR_LIST[i%len(COLOR_LIST)], path, args.legend_name_list[i])
        else:
            plot_learning_curve(data, COLOR_LIST[i%len(COLOR_LIST)], path, None)
    plt.xlabel('eopch')
    plt.ylabel('eval_score')
    plt.legend()
    plt.show()