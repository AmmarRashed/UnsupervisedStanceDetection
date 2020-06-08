import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from umap import UMAP


class Projector:
    DEFAULT_UMAP_PARAMS = dict(
        n_components=2,
        min_dist=0,
        n_neighbors=90,
        metric="cosine",
        random_state=42
    )
    DEFAULT_DIST_RANGE = [0.0, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99]
    DEFAULT_NEIGHBORS_RANGE = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    def __init__(self, vectors_path):
        self.vectors_path = vectors_path
        self.users, self.vectors, self.counts = self._load_vectors(vectors_path)

    @staticmethod
    def _load_vectors(vectors_path):
        file = np.load(vectors_path)
        users: np.ndarray = file['users']
        vectors: np.ndarray = file['vectors']
        counts: np.ndarray = file['counts']
        return users, vectors, counts

    @staticmethod
    def _project(vectors, **kwargs):
        return UMAP(**kwargs).fit_transform(vectors)

    def project(self, out_path, min_counts=3, **kwargs):
        params = self.DEFAULT_UMAP_PARAMS.copy()
        params.update(kwargs)

        ind = self.counts >= min_counts
        users = self.users[ind]
        vectors = self.vectors[ind]

        standard_embeddings = self._project(
            vectors=vectors,
            **params
        )
        np.savez(open(out_path, 'wb'),
                 umap=standard_embeddings, users=users)

    @staticmethod
    def plot_grid_search_heatmap(results, heatmap_destination="temp.png"):
        hm = pd.DataFrame(results)
        hm.index = hm.index.astype(int)
        hm = hm.sort_index(ascending=False)
        x = sorted(hm.columns)
        hm.index.name = "n_neighbors"
        hm.columns.name = "min_dist"

        sns.set(context='notebook', style='white', rc={'figure.figsize': (len(hm) * 2, len(hm.columns) * 2)},
                font_scale=2.5)

        sns.heatmap(hm[x], annot=True, cmap="Blues", annot_kws={"size": 30}, vmin=0.3, vmax=1, cbar=False)
        plt.savefig(heatmap_destination, bbox_inches='tight')

    @staticmethod
    def plot(embeddings, labels):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                              c=labels, s=0.1, cmap='Spectral')
        return scatter

    def grid_search(self, trials_dir, min_dists_range=DEFAULT_DIST_RANGE, n_neighbors_range=DEFAULT_NEIGHBORS_RANGE,
                    n_components=2,
                    metric="cosine", min_counts=3,
                    skip_existing=True, verbose=False):
        ind = self.counts >= min_counts
        users = self.users[ind]
        vectors = self.vectors[ind]

        umap_params = list(product(min_dists_range, n_neighbors_range))
        for min_dist, n in tqdm(umap_params, desc="UMAP"):
            if verbose:
                print(f"{min_dist}_{n}")
            out_path = os.path.join(trials_dir, f"{min_dist}_{n}.npz")
            if os.path.isfile(out_path) and skip_existing:
                continue

            standard_embeddings = self._project(
                vectors=vectors,
                random_state=42,
                n_components=n_components,
                n_neighbors=n,
                min_dist=min_dist,
                metric=metric
            )
            np.savez(open(out_path, 'wb'),
                     umap=standard_embeddings, users=users)
