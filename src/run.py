import numpy as np

from clustering import Clusterer, pickle
from projection import Projector, os

f = np.load("../TrElections.npz")

results, hm = Clusterer.cluster_projection_grid_search(
    f"../umap4_trials", users=f["users"], labels=f["labels"],
    # this means multiple clusters can be assigned the same label
    allow_multiple_clusters=True
)