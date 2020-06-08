import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_mutual_info_score as ami
from tqdm import tqdm


def correlate_clustering(df1, df2, metric_func, clusters_col="clusters", user_col="username", **kwargs):
    merged = pd.merge(df1[df1[clusters_col] >= 0], df2[df2[clusters_col] >= 0], on=user_col)
    y1, y2 = merged.labels_x, merged.labels_y
    return metric_func(y1, y2, **kwargs)


def calculate_alignment_matrix(dfs, metric_func, **kwargs):
    matrix = np.zeros((len(dfs), len(dfs)))
    for i, df1 in tqdm(enumerate(dfs)):
        for j, df2 in enumerate(dfs):
            matrix[i][j] = correlate_clustering(df1, df2, metric_func, **kwargs)
    return matrix


def plot_heatmap(frames, topics, func=ami):
    hm = calculate_alignment_matrix(frames, func)
    hm = pd.DataFrame(hm, columns=topics, index=topics).loc[reversed(topics)]
    fig = sns.heatmap(
        hm.round(2),
        annot=True,
        cmap="Blues",
        annot_kws={"size": 30},
        #         yticklabels=[i.title() for i in topics]
    )
    fig.set_yticklabels(labels=reversed(topics), rotation=45)
    fig.set_xticklabels(labels=topics, rotation=45)
    n = min(len(topics) * 2, 18)
    sns.set(context='notebook', style='white', rc={'figure.figsize': (n, n)}, font_scale=3.5)
    return fig


def mutual_information(topics, root="topicals"):
    frames = list()
    for topic in tqdm(topics):
        f = np.load(os.path.join(root, f"/{topic}.npz"))
        users = f["users"]
        clusters = f["clusters"]
        frames.append(pd.DataFrame({"users": users, "labels": clusters}))

    fig = plot_heatmap(frames, topics)