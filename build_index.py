import os
import argparse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import numpy as np
import re
from app.utils import _VECT_PATH

from bs4 import BeautifulSoup
import requests


def fetch_url(url, timeout=15):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for s in soup(["script", "style", "noscript", 
                  "meta", "link", "svg", "iframe"]):
        s.decompose()
    

    text = soup.get_text(" ", strip=True)
    text = re.sub(r'\\[u,x][a-f0-9]+', '', text)

    return text

def chunk_text(text, chunk_size_words=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunks.append(" ".join(words[i:i+chunk_size_words]))
    return chunks[:-1]

def build_index(urls_file, out_dir, chunk_size_words=300):
    os.makedirs(out_dir, exist_ok=True)
    texts = []
    metas = []

    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in tqdm(urls, desc="Fetch pages"):
        try:
            txt = fetch_url(url)
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue
        chunks = chunk_text(txt, chunk_size_words)
        for ch in chunks:
            texts.append(ch)
            metas.append(url)

    if not texts:
        raise RuntimeError("No texts fetched")

    vect = TfidfVectorizer(max_features=20000)
    X = vect.fit_transform(texts)

    n_components = min(256, X.shape[1] - 1, X.shape[0] - 1)
    n_components = max(n_components, 50)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)

    with open(_VECT_PATH, "wb") as f:
        pickle.dump({"vect": vect, "svd": svd}, f)

    embeddings = np.asarray(X_reduced, dtype="float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "metas": metas}, f)

    print("Index built:", out_dir)


if __name__ == "__main__":
    # p = argparse.ArgumentParser()
    # p.add_argument("--urls-file", required=True)
    # p.add_argument("--out-dir", required=True)
    # p.add_argument("--chunk-size", type=int, default=200)
    # args = p.parse_args()

    # build_index(args.urls_file, args.out_dir, args.chunk_size)
    build_index("urls.txt", "indexdata", 200)
