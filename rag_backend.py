from __future__ import annotations
from typing import List, Dict, Any
import os, requests, json

import chromadb
from sentence_transformers import SentenceTransformer

# --- Config: MUST match your index scripts ---
DB_PATH = "./chroma_store"
APTS_COLLECTION = "apartments"
STUDENTS_COLLECTION = "students"
EMBED_MODEL = "all-MiniLM-L6-v2"

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"  # LM Studio HTTP server

# --- 1. Load embedder (same model as embeddings) ---
embedder = SentenceTransformer(EMBED_MODEL)

def embed_query(text: str) -> List[List[float]]:
    return embedder.encode([text], normalize_embeddings=True).tolist()

# --- 2. Load Chroma persistent DB ---
client = chromadb.PersistentClient(path=DB_PATH)
apts = client.get_collection(APTS_COLLECTION)
students = client.get_collection(STUDENTS_COLLECTION)

# --- 3. LM Studio client ---
def run_llm(prompt: str) -> str:
    """Send prompt to local LM Studio model."""
    resp = requests.post(
        LMSTUDIO_URL,
        json={
            "model": "lmstudio",  # must match the model name in LM Studio UI
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 900,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# --- Format helpers ---
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
        if price not in ("", None, ""):
            header_parts.append(f"Price: €{price}")

        header = " | ".join(header_parts) if header_parts else "Apartment"
        lines.append(f"- {header}\n  {doc}")

    return "\n\n".join(lines) if lines else "No apartments found in the database."

def _format_student_context(res):
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    lines = []
    for doc, meta in zip(docs, metas):
        name = meta.get("name") or "Unnamed candidate"
        personality = meta.get("personality", "not specified")
        lifestyle = meta.get("lifestyle_summary", "not specified")
        sleep = meta.get("sleep_schedule", "not specified")
        noise = meta.get("noise_tolerance", "not specified")
        dog = meta.get("dog_friendliness", "not specified")
        clean = meta.get("cleanliness", "not specified")
        study = meta.get("study_habits", "not specified")
        chunk_idx = meta.get("chunk_index", "?")

        lines.append(
            f"- Candidate (chunk {chunk_idx}) — {name}\n"
            f"  Personality: {personality}\n"
            f"  Lifestyle: {lifestyle}\n"
            f"  Sleep schedule: {sleep}\n"
            f"  Noise tolerance: {noise}\n"
            f"  Dog friendliness: {dog}\n"
            f"  Cleanliness: {clean}\n"
            f"  Study habits: {study}\n"
            f"  Raw text: {doc[:200]}..."
        )

    return "\n\n".join(lines)

# --- 4. Main recommend() function ---
def recommend(query: str, n_apts: int = 3, n_students: int = 6) -> str:
    """
    Roommate-matching engine:
    - Ranks multiple candidate roommates
    - Computes compatibility matrix
    - Chooses best apartment + roommate pair
    """

    # --- Retrieve data from Chroma ---
    q_emb = embed_query(query)

    apt_res = apts.query(
        query_embeddings=q_emb,
        n_results=n_apts,
    )

    stu_res = students.query(
        query_embeddings=q_emb,
        n_results=n_students,
    )

    apt_context = _format_apartment_context(apt_res)
    stu_context = _format_student_context(stu_res)

    # --- LLM prompt ---
    prompt = f"""
You are an expert roommate-matching AI trained in lifestyle analysis, habit modeling,
interpersonal compatibility, and flat-sharing dynamics in university living.

USER PREFERENCES:
\"\"\"{query}\"\"\"

APARTMENTS AVAILABLE (TOP {n_apts} FROM RAG):
{apt_context}

POTENTIAL ROOMMATE CANDIDATES (TOP {n_students} FROM RAG):
{stu_context}

========================
TASK (STRICT RULES):
========================

### 1. Identify ALL candidates
For each candidate:
- Use the name from the metadata if present; otherwise keep the label as given.
- Provide 4–6 bullet points describing their lifestyle, habits, and personality.
- Summaries must ONLY use information from the context above. If unsure → say “not specified”.

### 2. Generate a compatibility matrix (0–100) for each candidate
Dimensions (each scored separately):
1. Cleanliness & tidiness  
2. Noise tolerance / quietness  
3. Social habits (introvert vs extrovert)  
4. Sleep schedule compatibility  
5. Dog friendliness / pet comfort  
6. Study/work habits  
7. Guest/party tolerance  
8. Boundary clarity  
9. Daily routine alignment  
10. Conflict-avoidance ability  

Output a small table-like block per candidate.

### 3. Compute an OVERALL SCORE (0–100)
Weighted average:
- cleanliness 15%
- noise 15%
- sleep schedule 15%
- boundaries 15%
- dog friendliness 10%
- social habits 10%
- study habits 10%
- routine alignment 5%
- conflict-avoidance 5%

### 4. Identify conflict risks for each roommate
- 2–4 risks MAX
- Each marked as: minor / moderate / serious

### 5. Produce a ranked list of roommates (best → worst)
Include 1–2 sentences explaining the ranking.

### 6. Choose ONE best apartment from the RAG results
Justify using:
- neighborhood
- price
- quiet vs social environment
- key amenities (bathroom, pets, etc.)
- only details that actually appear above

### 7. Final output format:

#### 🧑‍🤝‍🧑 Best Roommate Match
(name or label)

#### 🔢 Compatibility Score: XX/100

#### 📊 Compatibility Matrix
(list the 10 scores clearly)

#### ⚠️ Conflict Risks
- Risk (severity)

#### 🏠 Best Apartment Match
- Apartment title
- Why it fits both the user AND the chosen roommate

#### 🟣 Final Verdict (≤ 4 sentences)
Give a clear recommendation.

STRICT RULE:
You MUST NOT invent new facts not present in the apartment or student context.
If a detail is unknown, say “not specified”.
"""

    return run_llm(prompt)

# --- Manual test ---
if __name__ == "__main__":
    q = (
        "I want a quiet big dog friendly apartment in barrio Salamanca with a budget of "
        "3300 euros a month max. I want my own private bathroom and sunset views. "
        "I am very extroverted, not very organised and spontaneous, and I cannot handle "
        "people that are too structured."
    )
    print(recommend(q))
