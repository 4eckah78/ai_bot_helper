from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle
import os

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
_VECT_PATH = os.path.join(_CACHE_DIR, "vect_svd.pkl")


def _train_vectorizer_and_svd(corpus, n_components=256):
    vect = TfidfVectorizer(max_features=20000)
    X = vect.fit_transform(corpus)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit_transform(X)
    with open(_VECT_PATH, "wb") as f:
        pickle.dump({"vect": vect, "svd": svd}, f)
    return vect, svd


def _load_vectorizer_and_svd():
    if os.path.exists(_VECT_PATH):
        import pickle

        with open(_VECT_PATH, "rb") as f:
            d = pickle.load(f)
        return d["vect"], d["svd"]
    return None, None


def get_embedding(text):
    vect, svd = _load_vectorizer_and_svd()
    if vect is None or svd is None:
        vect, svd = _train_vectorizer_and_svd(
            [text], n_components=min(256, max(32, len(text)))
        )
    X = vect.transform([text])
    vec = svd.transform(X)
    vec = np.asarray(vec).reshape(-1)
    return vec.astype("float32")


