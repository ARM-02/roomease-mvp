from __future__ import annotations
import os
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# --- Config: MUST match your embeddings script ---
DB_PATH = "./chroma_store"
APTS_COLLECTION = "apartments"
STUDENTS_COLLECTION = "students"
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- 1. Load embedder (same model as embeddings.py) ---
embedder = SentenceTransformer(EMBED_MODEL)

def embed_query(text: str) -> List[List[float]]:
    # use normalized embeddings because you stored normalized ones in Chroma
    return embedder.encode([text], normalize_embeddings=True).tolist()

# --- 2. Load Chroma persistent DB ---
client = chromadb.PersistentClient(path=DB_PATH)
apts = client.get_collection(APTS_COLLECTION)
students = client.get_collection(STUDENTS_COLLECTION)

llm = Llama(
    model_path="models/qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=50,   # M1 metal GPU acceleration
    temperature=0.3,
)


def _format_apartment_context(res: Dict[str, Any]) -> str:
    """Pretty-print top apartment hits for the prompt."""
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    lines = []
    for doc, meta in zip(docs, metas):
        title = str(
            meta.get("title")
            or meta.get("Title")
            or meta.get("name")
            or meta.get("Name")
            or ""
        ).strip()

        hood = str(
            meta.get("neighborhood")
            or meta.get("Neighborhood")
            or ""
        ).strip()

        price = meta.get("price") or meta.get("Price") or ""
        header_parts = []
        if title:
            header_parts.append(title)
        if hood:
            header_parts.append(f"Neighborhood: {hood}")
        if price not in ("", None):
            header_parts.append(f"Price: €{price}")

        header = " | ".join(header_parts) if header_parts else "Apartment"
        lines.append(f"- {header}\n  {doc}")

    return "\n\n".join(lines) if lines else "No apartments found in the database."

def _format_student_context(res: Dict[str, Any]) -> str:
    """Pretty-print student chunks as 'similar lifestyle examples'."""
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    lines = []
    for doc, meta in zip(docs, metas):
        chunk_idx = meta.get("chunk_index", "?")
        lines.append(f"- Example student answer (chunk {chunk_idx}): {doc}")

    return "\n\n".join(lines) if lines else "No similar student responses found."

# --- 4. Main recommend() function using both collections ---
def recommend(query: str, n_apts: int = 3, n_students: int = 3) -> str:
    """
    Given a free-text description of what the student wants,
    retrieve relevant apartments + similar student responses,
    then ask Mistral to pick + justify the best option.
    """

    # 1) Embed query
    q_emb = embed_query(query)

    # 2) Retrieve apartments
    apt_res = apts.query(
        query_embeddings=q_emb,
        n_results=n_apts,
    )

    # 3) Retrieve student lifestyle chunks
    stu_res = students.query(
        query_embeddings=q_emb,
        n_results=n_students,
    )

    apt_context = _format_apartment_context(apt_res)
    stu_context = _format_student_context(stu_res)

    # 4) Build prompt for Mistral
    prompt = f"""
You are an intelligent housing assistant helping a university student in Madrid
find a suitable apartment and flat-share environment.

User preferences (their own description):
\"\"\"{query}\"\"\"

Top apartment options from the database:
{apt_context}

Example student responses with similar lifestyle and preferences:
{stu_context}

TASK:
1. Choose the ONE apartment that best fits the user.
2. Justify your choice using concrete details: neighborhood, price, quiet vs social, proximity to study needs, etc.
3. Briefly infer what kind of roommate dynamic this suggests (quiet, social, party, very organized, etc.).
4. Be specific and practical so the user could actually decide based on your answer.
5. Keep the answer under 250 words.

Now provide your recommendation:
"""

    # 5) Call local Mistral
    resp = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )
    print(resp["choices"][0]["message"]["content"])


# --- 5. Quick manual test ---
if __name__ == "__main__":
    q = "I want a quiet big dog friendly apartment in barrio salamanca with a budget of 3300 euros a month a max, i want my own private bathroom and sunset views"
    print(recommend(q))
