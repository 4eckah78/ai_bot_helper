import os
import pickle

import faiss
import numpy as np


class Retriever:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.index = None
        self.meta = None
        self._load()

    def _load(self):
        idx_path = os.path.join(self.index_dir, "faiss.index")
        meta_path = os.path.join(self.index_dir, "meta.pkl")
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise RuntimeError("Index or meta not found. Build index first.")
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            d = pickle.load(f)
        self.texts = d["texts"]
        self.metas = d["metas"]

    def query(self, q_vector, top_k=5):
        # q_vector: np.array (dim,)
        v = q_vector.astype("float32").reshape(1, -1)
        norm = np.linalg.norm(v)
        if norm == 0:
            norm = 1.0
        v = v / norm
        D, I = self.index.search(v, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append(
                {"score": float(score), "text": self.texts[idx], "url": self.metas[idx]}
            )
        return results
