# Embedding-based Unsupervised User Stance Detection
- paper: [Embeddings-Based Clustering for Target Specific Stances: The Case of a Polarized Turkey
](https://arxiv.org/abs/2005.09649)
- We propose an unsupervised user stance detection method to capture fine grained divergences in a community across various topics. We employ pre-trained universal sentence encoders to represent users based on the content of their tweets on a particular topic. User vectors are projected into a lower dimensional space using UMAP, then clustered using HDBSCAN.
### Fine-grain stances
- Our method is able to capture stances to the party-affiliation level in a completely unsupervised manner.
![](ed.png)
### Mutual information
- Given the resultant user stances, we are able to observe correlations between topics and compute topic polarization.
 ![](ami.png)
 ### Semantic divergence between clusters
 - We identify the most prominent terms in
each cluster to show how people talk about the same issue
in different contexts.
 ![](wc.png)
#### Requirements
Note: This work was tested using umap-learn 0.3.x. Newer versions might not work as expected.
- [umap-learn 0.3.x](https://pypi.org/project/umap-learn/0.3.10/)
- [hdbscan 0.8.x](https://pypi.org/project/hdbscan/)
- [tensorflow-hub 0.8.x](https://pypi.org/project/tensorflow-hub/)
- [tensorflow-text 2.2.x](https://pypi.org/project/tensorflow-text/)