import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


import faiss
import pickle
import numpy as np
import os

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
            results.append({"score": float(score), "text": self.texts[idx], "url": self.metas[idx]})
        return results
    
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
        vect, svd = _train_vectorizer_and_svd([text], n_components=min(256, max(32, len(text))))
    X = vect.transform([text])
    vec = svd.transform(X)
    vec = np.asarray(vec).reshape(-1)
    print("TFIDF features:", X.shape[1], " SVD dim:", svd.n_components)
    return vec.astype("float32")

from gigachat import GigaChat


class GigachatClient:
    def __init__(self, auth_key: str):
        if not auth_key:
            raise RuntimeError("GIGACHAT_AUTH_KEY required")
        self.auth = auth_key

    def generate(self, prompt: str, model="GigaChat", max_tokens=800, temperature=0.3):
        messages = [
            {"role": "system", "content": "Ты помощник, отвечай строго по контексту, цитируй номера источников [n]."},
            {"role": "user", "content": prompt}
        ]
        giga = GigaChat(credentials=self.auth)
        response = giga.chat({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        return response.choices[0].message.content


# from .retriever import Retriever
# from .utils import get_embedding
# from .gigachat_client import GigachatClient

INDEX_DIR = os.getenv("INDEX_DIR", "indexdata")
GIGACHAT_KEY = os.getenv("GIGACHAT_AUTH_KEY")

if not GIGACHAT_KEY:
    raise RuntimeError("Set GIGACHAT_AUTH_KEY env var")

retriever = Retriever(INDEX_DIR)
gclient = GigachatClient(GIGACHAT_KEY)

app = FastAPI(title="EORA RAG with GigaChat")

class Query(BaseModel):
    question: str
    top_k: int = 3

class Answer(BaseModel):
    answer: str
    used_urls: List[str]

@app.post("/answer", response_model=Answer)
def answer(q: Query):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    
    qvec = get_embedding(q.question)
    hits = retriever.query(qvec, top_k=q.top_k)

    url_to_idx = {}
    parts = []
    for h in hits:
        url = h["url"]
        if url not in url_to_idx:
            url_to_idx[url] = len(url_to_idx) + 1
        idx = url_to_idx[url]

        snippet = h["text"]
        parts.append(f"[{idx}] {snippet}")

    context = "\n\n".join(parts)
    prompt = f"Используй только следующий контекст с пометками [n]:\n\n{context}\n\nВопрос: {q.question}\nОтвечай кратко и после фактов указывай номера источников в квадратных скобках и ссылки на источники (inline-ссылки как в markdown)."

    resp_text = gclient.generate(prompt)

    used_idxs = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", resp_text)]
    idx_to_url = {v:k for k,v in url_to_idx.items()}
    used_urls = []
    for i in used_idxs:
        u = idx_to_url.get(i)
        if u and u not in used_urls:
            used_urls.append(u)

    return Answer(answer=resp_text, used_urls=used_urls)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)