import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .retriever import Retriever
from .gigachat_client import GigachatClient
from .utils import get_embedding


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
        parts.append(f"{snippet} [[{idx}]({url})]")

    context = "\n\n".join(parts)

    prompt = (
        f"Используй только следующий контекст:\n\n{context}\n\nВопрос: {q.question}\n\n"
        f"Только после (не перед) каждого факта ставь его номер и ссылку вот так - [[n](ссылка)]"
    )
    print(prompt)

    resp_text = gclient.generate(prompt)

    return Answer(answer=resp_text)