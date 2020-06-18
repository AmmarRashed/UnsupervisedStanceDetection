import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from wordcloud import STOPWORDS, WordCloud


def get_word_counts(file, text_col=None):
    res = {}
    tfg = 0
    pbar = tqdm(desc=file.split("/")[-1])
    with open(file) as f:
        for i, l in enumerate(f, 1):
            l = l.replace('.', '').strip().lower()
            if text_col is not None:
                l = l.split('\t')[text_col]
            for w in l.split():
                if len(w) <= 2:
                    continue
                res.setdefault(w, 0)
                res[w] += 1
                tfg += 1
            if i % 10_000 == 0:
                pbar.update(i)
    return res, tfg


def count_words_csv(text_series):
    counter = Counter()
    text_series.apply(lambda x: counter.update(x.lower().strip().split()))
    return dict(counter), sum(counter.values())


def valence_step(tfe1, tfg1, tfe2, tfg2, out, e):
    a = tfe1 / tfg1
    b = tfe2 / tfg2
    v1 = 2 * (a / (a + b)) - 1
    if v1 >= 0.8:
        out.write(f"{v1 * np.log(tfe1)}\t{e}\t{v1}\t{tfe1}\n")


def sort_scores(file):
    pd.read_csv(
        file, sep='\t', names=["score", "term", "valence", "frequency"]
    ).sort_values(
        "score", ascending=False
    ).to_csv(
        file.replace("txt", "csv"), sep='\t', index=None
    )
    os.remove(file)


def valence(tf1, tfg1, tf2, tfg2, out):
    with open(out, 'w') as o:
        Parallel(n_jobs=-1, backend='threading')(
            delayed(valence_step)(
                tfe, tfg1, 0 if e not in tf2 else tf2[e], tfg2, o, e
            ) for e, tfe in tf1.items() if len(e) > 2
        )
    print("Sorting terms")
    sort_scores(out)


def pipeline(df1, df2, out1, out2=None, text_col='text'):
    print("Counting terms...")
    (tf1, tfg1), (tf2, tfg2) = Parallel(n_jobs=2, backend='threading')(
        delayed(count_words_csv)(df[text_col]) for df in [df1, df2])
    del df1, df2
    print("Calculating valence for group 1 ...")
    valence(tf1, tfg1, tf2, tfg2, out1)
    if out2 is not None:
        print("Calculating valence for group 2 ...")
        valence(tf2, tfg2, tf1, tfg1, out2)


def plot_worcloud(file, mask_path):
    params = dict(width=800, height=800,
                  background_color='white',
                  stopwords=set(STOPWORDS),
                  min_font_size=10)
    if mask_path is not None:
        params["mask"] = np.array(Image.open(mask_path))
    wordcloud = WordCloud(**params)

    scores = pd.read_csv(file, sep='\t')[:500].set_index("term").to_dict()["score"]
    fig = wordcloud.generate_from_frequencies(scores)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{file}.png")


def calculate_top_terms(clusters_path, tweets_path, prefix, user_col, text_col, use_clusters=True, mask_path=None):
    enf = np.load(clusters_path)
    df = pd.read_pickle(tweets_path)
    users, clusters = enf["users"], enf["clusters"]
    if use_clusters:
        labels = dict(zip(users, clusters))
        ind = clusters >= 0
    else:
        y = np.array(
            [1 if re.search("(lfc)|(liverpool)", x.lower()) else 0 if re.search("(cfc)|(chelsea)", x.lower()) else -1
             for x in enf["users"]])
        ind = y >= 0
        labels = dict(zip(users, y))
    df = df[df[user_col].apply(lambda x: x in labels)]
    df = df.assign(label=df[user_col].apply(lambda x: labels[x]))

    o1 = os.path.join("terms", f"{prefix}.0.txt")
    o2 = os.path.join("terms", f"{prefix}.1.txt")
    pipeline(df[df.label == 0], df[df.label == 1],
             out1=o1,
             out2=o2,
             text_col=text_col
             )

    for o in enumerate(o1, o2):
        plot_worcloud(o, mask_path=mask_path)
