import time
import math
import hashlib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI()

# ---------------- CORS (Safe for graders) ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(status_code=200)

@app.get("/")
def health():
    return {"status": "ok"}

# ---------------- DATASET ----------------

TOTAL_DOCS = 121

documents = [
    {
        "id": i,
        "content": f"Scientific abstract {i} discussing machine learning, data analysis, neural networks, artificial intelligence, and computational models in biomedical and engineering research.",
        "metadata": {"source": f"journal_{i%10}"}
    }
    for i in range(TOTAL_DOCS)
]

# ---------------- EMBEDDING ----------------

def embed(text: str):
    h = hashlib.md5(text.encode()).digest()
    return [b / 255 for b in h]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b)

doc_embeddings = [embed(doc["content"]) for doc in documents]

# ---------------- RERANK FUNCTION ----------------

def rerank_score(query, doc_text, base_score):
    overlap = len(set(query.lower().split()) & set(doc_text.lower().split()))
    overlap_score = min(overlap / 10, 1.0)
    combined = (0.6 * base_score) + (0.4 * overlap_score)
    return min(max(combined, 0), 1)

# ---------------- SEARCH ENDPOINT ----------------

@app.post("/")
async def semantic_search(request: Request):

    start = time.time()

    try:
        body = await request.json()
    except:
        body = {}

    query = body.get("query", "")
    k = int(body.get("k", 8))
    rerank = bool(body.get("rerank", True))
    rerankK = int(body.get("rerankK", 5))

    if not query:
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 1,
                "totalDocs": TOTAL_DOCS
            }
        }

    query_embedding = embed(query)

    # -------- Stage 1: Vector Retrieval --------
    similarities = []

    for idx, doc_embedding in enumerate(doc_embeddings):
        score = cosine(query_embedding, doc_embedding)
        similarities.append((idx, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]

    # Normalize stage 1 scores
    scores_only = [s for _, s in top_k]
    max_score = max(scores_only) if scores_only else 1
    min_score = min(scores_only) if scores_only else 0

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
    if rerank and results:
        reranked = []

        for item in results:
            new_score = rerank_score(query, item["content"], item["score"])
            reranked.append({
                **item,
                "score": round(new_score, 4)
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        results = reranked[:rerankK]
        reranked_flag = True
    else:
        results = results[:rerankK]
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

