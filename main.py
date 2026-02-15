import time
import math
import hashlib
import random
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---------------- CONFIG ----------------

TOTAL_DOCS = 121

# ---------------- DATASET (Mock Scientific Abstracts) ----------------

documents = [
    {
        "id": i,
        "content": f"Scientific abstract {i} discussing machine learning, data analysis, neural networks, artificial intelligence, and computational models in biomedical and engineering research.",
        "metadata": {"source": f"journal_{i%10}"}
    }
    for i in range(TOTAL_DOCS)
]

# ---------------- EMBEDDING (Lightweight Deterministic) ----------------

def embed(text: str):
    h = hashlib.md5(text.encode()).digest()
    return [b / 255 for b in h]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot / (norm_a * norm_b)

# Precompute document embeddings (cached)
doc_embeddings = [embed(doc["content"]) for doc in documents]

# ---------------- REQUEST MODEL ----------------

class SearchRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    rerankK: int = 5

# ---------------- RE-RANK FUNCTION ----------------

def rerank_score(query: str, doc_text: str, base_score: float):
    # Simulated LLM-like relevance scoring
    overlap = len(set(query.lower().split()) & set(doc_text.lower().split()))
    overlap_score = min(overlap / 10, 1.0)
    combined = (0.6 * base_score) + (0.4 * overlap_score)
    return min(max(combined, 0), 1)

# ---------------- SEARCH ENDPOINT ----------------

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/")
def semantic_search(request: SearchRequest):

    start = time.time()

    if not request.query:
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 1,
                "totalDocs": TOTAL_DOCS
            }
        }

    query_embedding = embed(request.query)

    # -------- Stage 1: Vector Retrieval --------
    similarities = []

    for idx, doc_embedding in enumerate(doc_embeddings):
        score = cosine(query_embedding, doc_embedding)
        similarities.append((idx, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:request.k]

    # Normalize stage 1 scores
    max_score = max([s for _, s in top_k]) if top_k else 1
    min_score = min([s for _, s in top_k]) if top_k else 0

    results = []

    for idx, score in top_k:
        norm_score = (score - min_score) / (max_score - min_score + 1e-9)
        results.append({
            "id": documents[idx]["id"],
            "score": round(norm_score, 4),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    # -------- Stage 2: Re-ranking --------
    if request.rerank and results:
        reranked = []

        for item in results:
            new_score = rerank_score(
                request.query,
                item["content"],
                item["score"]
            )
            reranked.append({
                **item,
                "score": round(new_score, 4)
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        results = reranked[:request.rerankK]

        reranked_flag = True
    else:
        results = results[:request.rerankK]
        reranked_flag = False

    latency = max(1, int((time.time() - start) * 1000))

    return {
        "results": results,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": TOTAL_DOCS
        }
    }

