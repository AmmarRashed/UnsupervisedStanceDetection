import numpy as np
import pandas as pd
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text
from tqdm import tqdm


class Encoder:
    DEFAULT_MODEL = "https://tfhub.dev/google/universal-sentence-encoder/4"

    def __init__(self, model_url: str = DEFAULT_MODEL):
        """
        Args:
            model_url: str, url to the Universal Sentence Encoder model
            Default is English USE >> https://tfhub.dev/google/universal-sentence-encoder/4
            for the multilingual version, use: https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
            more models are available at: https://tfhub.dev/google/collections/universal-sentence-encoder/1
        """
        self.model_url = model_url
        self.encoder = self._load_model()

    def _load_model(self):
        return hub.load(self.model_url)

    def encode(self, text):
        return np.array(self.encoder(text))

    def encode_df(self, df: pd.DataFrame, out_path: str, user_col: str = "username", text_col: str = "text"):
        users = list()
        vectors = list()
        counts = list()

        for user, tweets in tqdm(df.groupby(user_col)[text_col]):
            try:
                vs = np.array(self.encoder(tweets.tolist()))
                users.append(user)
                vectors.append(np.mean(vs, axis=0))
                counts.append(len(tweets))
            except Exception as e:
                print(user)
                print(e)

        np.savez(out_path, users=np.array(users), vectors=np.array(vectors), counts=np.array(counts),
                 allow_pickle=True)


class EncoderBERT(Encoder):
    DEFAULT_MODEL = "roberta-base-nli-stsb-mean-tokens"

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_url)
