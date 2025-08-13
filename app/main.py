import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .gigachat_client import GigachatClient
from .retriever import Retriever
from .utils import get_embedding, replace_refs

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

    idx = 0
    urls = []
    parts = []
    for h in hits:
        url = h["url"]
        if url not in urls:
            idx += 1
        urls.append(url)

        snippet = h["text"]
        parts.append(f"[{idx}] {snippet} ")

    context = "\n\n".join(parts)

    prompt = (
        f"Используй ТОЛЬКО предоставленный контекст. Контекст (каждый фрагмент помечен [n]):\n\n"
        f"{context}\n\n"
        f"Вопрос: {q.question}\n\n"
        f"Отвечай кратко и включай в ответ метки [n] там, где ты использовал конкретный фрагмент."
    )

    resp_text = gclient.generate(prompt)

    resp_text = replace_refs(resp_text, urls)

    return Answer(answer=resp_text)


