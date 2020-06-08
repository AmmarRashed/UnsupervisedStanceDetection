###############################################################################
#   Code written by Ammar Rashid (Özyeğin University)
#   ammar.rasid@ozu.edu.tr
#   and modified by Kareem Darwish (Qatar Computing Research Institute)
#   kdarwish@hbku.edu.qa
#   The code is provided for research purposes ONLY
###############################################################################

###############################################################################
# sys.argv[1] is a tab separated file with first column containing UserIDs
# and second column containing tweets
###############################################################################
# there are many options for the universal sentence encoder including multilingual
# models, Transformer model (slow), and CNN model (fast)
# check out: https://tfhub.dev/google/universal-sentence-encoder/1
# for options
###############################################################################

import ntpath
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from hdbscan import HDBSCAN
from tqdm import tqdm
from umap import UMAP


def cluster_users(df, encoder: Callable, min_tweets=3, user_col="username",
                  tweet_col="norm_tweet", save_at="temp.npz",
                  min_dist=0.0, n_neighbors=90, **kwargs):
    gs = df.groupby(user_col)
    users = list()
    vectors = list()
    for user, frame in tqdm(gs):
        if len(frame) < min_tweets:
            continue
        try:
            tweets = frame[tweet_col]
            vec = np.mean(np.array(encoder(tweets.tolist())), axis=0)
            users.append(user)
            vectors.append(vec)
        except Exception as e:
            print(f"ERROR at:{user}")
            print(e)
            print()

    users: np.ndarray = users
    vectors: np.ndarray = vectors

    standard_embeddings = UMAP(
        random_state=42,
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine', **kwargs
    ).fit_transform(vectors)
    print("Projection complete")

    params = dict()

    clusterer = cluster_embeddings(standard_embeddings, **kwargs)
    params['clusters'] = clusterer.labels_
    params["allow_pickle"] = True
    np.savez(open(save_at + '.cluster', 'wb'), users=np.array(users), vectors=np.array(vectors),
             umap=np.array(standard_embeddings), clusters=np.array(clusterer.labels_))

    output_file = open(save_at + '.clusters.txt', mode='w')
    for i in range(len(clusterer.labels_)):
        output_file.write(str(users[i]) + '\t' + str(clusterer.labels_[i]) + '\n')
    output_file.close()


def plot_clusters_no_labels(embeddings_path, clusters_col="clusters", green_label="pro", red_label='anti', align=False,
                            title=None, include_ratio=True, labeled_only=False):
    if title is None:
        title = ntpath.basename(embeddings_path).split('.')[0]
    f = np.load(embeddings_path)
    users = f["users"]
    clusters = f[clusters_col]
    cluster_ratio = round(sum(clusters >= 0) * 100 / len(clusters), 2)
    em = f["umap"]

    ind = clusters >= 0
    users = users[ind]
    clusters = clusters[ind]
    em = em[ind, :]
    c = ['red', 'blue', 'green', 'black', 'orange', 'teal']
    if align:
        d = align_clusters_with_labels(
            pd.DataFrame({"username": users, "clusters": clusters})
        )
        c = ['red', 'blue', 'green', 'black', 'orange', 'teal', 'olive', 'yellow']
    else:
        labels_dict = {}

    cmap = list()
    for i in range(len(clusters)):
        cmap.append(c[clusters[i] - 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = plt.scatter(em[:, 0], em[:, 1], c=cmap,
                          s=0.5, cmap='Spectral')
    ax.set_title(title, fontsize=22)
    plt.show()
    return scatter


def align_clusters_with_labels(df, allow_multiple_clusters=True):
    df = df[df.clusters >= 0]
    g = df.groupby(["label", "clusters"]).count().sort_values("username", ascending=False)

    d = {}
    while len(g) > 0:
        label, cluster = g.index[0]
        d[cluster] = label
        g = g.reset_index()
        g = g[(g.label != label) & (g.clusters != cluster)] \
            .set_index(["label", "clusters"]) \
            .sort_values("username", ascending=False)
    unlabeled_clusters = set(df.clusters) - set(d.keys())
    if allow_multiple_clusters and len(unlabeled_clusters) > 0:
        g = df.groupby(["label", "clusters"]).count().sort_values("username", ascending=False).reset_index()
        for c in unlabeled_clusters:
            l = g.set_index("clusters").loc[c].label
            if isinstance(l, pd.Series):
                l = l.iloc[0]
            d[c] = l

            g = g[g.clusters != c]

    return d


def cluster_embeddings(standard_embedding,
                       min_cluster_size=None,
                       min_samples=None,
                       plot_tree=False,
                       min_samples_div=1000,
                       min_cluster_size_div=100,
                       **kwargs):
    if min_cluster_size is None:
        min_cluster_size = max(10, len(standard_embedding) // min_cluster_size_div)
    if min_samples is None:
        min_samples = max(10, len(standard_embedding) // min_samples_div)
    clusterer = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size, **kwargs
    ).fit(standard_embedding)
    if plot_tree:
        clusterer.condensed_tree_.plot()
    # return clusterer.labels_, clusterer.condensed_tree_
    return clusterer


if __name__ == "__main__":
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')  # You can use different encoders here

    inputFile = sys.argv[1]  # ex. trump.tsv
    df_text = pd.read_csv(inputFile, header=None, usecols=[0, 1], error_bad_lines=False, sep='\t')
    df_text.columns = ['User', 'Text']
    df_text = df_text.apply(lambda s: s.str.strip())
    cluster_users(df_text, embed, user_col='User', tweet_col='Text', save_at=inputFile + '.npz')
    plot_clusters_no_labels(inputFile + '.npz.cluster')
