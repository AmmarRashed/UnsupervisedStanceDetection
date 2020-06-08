import os
import pickle
from typing import Optional

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from projection import Projector


class Clusterer:

    def __init__(self, projection_path):
        self.projection_path = projection_path
        self._params = self._load_standard_embeddings()
        self.N: int = len(self._params["users"])

    def _load_standard_embeddings(self):
        file = np.load(self.projection_path, allow_pickle=True)
        params = dict()
        for k in file.keys():
            params[k] = file[k]
        return params

    @staticmethod
    def _cluster(standard_embeddings, **kwargs):
        return hdbscan.HDBSCAN(**kwargs).fit(standard_embeddings)

    def cluster(self, min_samples: Optional[int] = None, min_cluster_size: Optional[int] = None,
                min_samples_divisor: int = 1000, min_cluster_size_divisor: int = 100,
                tree_path=None,
                **kwargs):
        if min_samples is None:
            kwargs["min_samples"] = max(10, self.N // min_samples_divisor)
        if min_cluster_size is None:
            kwargs["min_cluster_size"] = max(10, self.N // min_cluster_size_divisor)

        model = self._cluster(standard_embeddings=self._params["umap"],
                              **kwargs
                              )

        self._params["clusters"] = model.labels_
        np.savez(open(self.projection_path, 'wb'), **self._params)
        if tree_path is not None:
            pickle.dump(model.condensed_tree_, open(tree_path, 'wb'), protocol=3)

    @staticmethod
    def plot_tree(path):
        sns.set(context='notebook', style='white', rc={'figure.figsize': (15, 10)})
        return pickle.load(open(path, 'rb')).plot()

    def plot(self, labels_col="clusters"):
        return Projector.plot(embeddings=self._params["umap"], labels=self._params[labels_col])

    def inject_labels(self, users, labels):
        labels_dict = dict(zip(users, labels))
        self._params["labels"] = np.array(
            [labels_dict[u] if u in labels_dict else 'unk' for u in self._params["users"]]
        )

    def align_clusters_with_labels(self, allow_multiple_clusters=True):
        labels = self._params["labels"]
        ind = labels != 'unk'
        users = self._params["users"][ind]
        labels = labels[ind]

        df = pd.DataFrame(
            {"username": users, "labels": labels}
        ).merge(
            pd.DataFrame({"username": self._params["users"], "clusters": self._params["clusters"]})
        )

        g = df.groupby(["label", "clusters"]).count().sort_values("username", ascending=False)

        d = {}
        while len(g) > 0:
            label, cluster = g.index[0]
            d[cluster] = label
            g = g.reset_index()
            g = g[(g.label != label) & (g.clusters != cluster)].set_index(["label", "clusters"]).sort_values("username",
                                                                                                             ascending=False)
        unlabeled_clusters = set(df.clusters) - set(d.keys())
        if allow_multiple_clusters and len(unlabeled_clusters) > 0:
            g = df.groupby(["label", "clusters"]).count().sort_values("username", ascending=False).reset_index()
            for c in unlabeled_clusters:
                l = g.set_index("clusters").loc[c].label
                if isinstance(l, pd.Series):
                    l = l.iloc[0]
                d[c] = l

                g = g[g.clusters != c]

        self._params["predictions"] = np.array([d[x] if x in d else 'unk' for x in self._params['clusters']])

    def evaluate(self, metric=f1_score, report=True):
        if "predictions" not in self._params:
            raise Exception("No labels aligned with clusters")

        y = self._params["labels"]
        p = self._params["predictions"]

        ind = y != 'unk'
        y = y[ind]
        p = p[ind]

        s = set(y)
        if report:
            return pd.DataFrame(classification_report(y, p, labels=s, output_dict=True))

        return metric(y, p, labels=s, average='micro')

    @staticmethod
    def cluster_projection_grid_search(trials_dir, users=None, labels=None, allow_multiple_clusters=True):
        results = dict()
        for fn in tqdm(os.listdir(trials_dir)):
            if not fn.endswith("npz"):
                continue
            min_dist, n_neighbors = fn.replace(".npz", '').split("_")
            projection_path = os.path.join(trials_dir, fn)
            c = Clusterer(projection_path)
            c.cluster()
            # title = f"min_dist:{min_dist}\tn_neighbors:{n_neighbors}".expandtabs()
            plot_path = os.path.join(trials_dir, f"{min_dist}_{n_neighbors}.png")
            c.inject_labels(users=users, labels=labels)
            c.align_clusters_with_labels(allow_multiple_clusters=allow_multiple_clusters)
            fig = c.plot()
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

            score = c.evaluate()
            results.setdefault(min_dist, dict())
            results[min_dist][n_neighbors] = score
        return results
