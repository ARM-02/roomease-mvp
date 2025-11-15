from __future__ import annotations
import os
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq   # <<--- NEW


# --- Config: MUST match your embeddings script ---
DB_PATH = "./chroma_store"
APTS_COLLECTION = "apartments"
STUDENTS_COLLECTION = "students"
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- 1. Load embedder (same model as embeddings.py) ---
embedder = SentenceTransformer(EMBED_MODEL)

def embed_query(text: str) -> List[List[float]]:
    return embedder.encode([text], normalize_embeddings=True).tolist()

# --- 2. Load Chroma persistent DB ---
client = chromadb.PersistentClient(path=DB_PATH)
apts = client.get_collection(APTS_COLLECTION)
students = client.get_collection(STUDENTS_COLLECTION)

# --- 3. Groq client (your API key directly here) ---
GROQ_API_KEY = "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"

groq_client = Groq(api_key=GROQ_API_KEY)

def run_llm(prompt: str) -> str:
    """Send prompt to Groq Llama 3 model."""
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.2,
    )
    return completion.choices[0].message.content

# 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

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

def _format_student_context(res):
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    lines = []
    for doc, meta in zip(docs, metas):
        name = meta.get("name", "Unnamed")
        personality = meta.get("personality", "")
        lifestyle = meta.get("lifestyle_summary", "")
        sleep = meta.get("sleep_schedule", "")
        noise = meta.get("noise_tolerance", "")

        lines.append(
            f"- {name}\n"
            f"  Personality: {personality}\n"
            f"  Lifestyle: {lifestyle}\n"
            f"  Sleep schedule: {sleep}\n"
            f"  Noise tolerance: {noise}\n"
            f"  Raw text: {doc[:200]}..."
        )

    return "\n\n".join(lines)


# --- 4. Main recommend() function using both collections ---
def recommend(query: str, n_apts: int = 3, n_students: int = 6) -> str:
    """
    Upgraded roommate-matching engine:
    - Ranks multiple candidate roommates
    - Creates structured summaries
    - Computes a compatibility matrix (0–100)
    - Identifies conflict risks with severity score
    - Recommends final apartment + roommate pair
    """

    # --- Retrieve data ---
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

    # Format roommate context
    roommate_context = []
    for doc, meta in zip(stu_res["documents"][0], stu_res["metadatas"][0]):
        roommate_context.append(
            {
                "chunk": meta.get("chunk_index"),
                "text": doc,
                "source": meta.get("source"),
            }
        )

    # Convert to readable block
    roommate_block = "\n\n".join(
        [
            f"[Candidate {i+1}] (chunk {c['chunk']})\n{c['text']}"
            for i, c in enumerate(roommate_context)
        ]
    )

    # --- FULL LLM PROMPT ---
    prompt = f"""
You are an expert roommate-matching AI trained in lifestyle analysis, habit modeling,
interpersonal compatibility, and flat-sharing dynamics in university living.

USER PREFERENCES:
\"\"\"{query}\"\"\"


APARTMENTS AVAILABLE (TOP {n_apts} FROM RAG):
{apt_context}


POTENTIAL ROOMMATE CANDIDATES (TOP {n_students} FROM PDF):
{roommate_block}


========================
TASK (STRICT RULES):
========================

### 1. Identify **ALL** candidates
For each candidate:
- Extract ANY name mentioned (if present).
- If no name: assign a label like “Candidate A – quiet outgoing student”.
- Provide **4–6 bullet points** describing their lifestyle, habits, and personality.
- Summaries must ONLY use information from the chunk. If unsure → say “not specified”.

### 2. Generate a **compatibility matrix (0–100)** for each candidate
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

Output must be a table-like block per candidate.

### 3. Compute an OVERALL SCORE (0–100)
Use a weighted average:
- cleanliness 15%
- noise 15%
- sleep schedule 15%
- boundaries 15%
- dog friendliness 10%
- social habits 10%
- study habits 10%
- routine alignment 5%
- conflict-avoidance 5%

### 4. Identify **conflict risks** for each roommate
- 2–4 risks MAX
- Each marked as: minor / moderate / serious

### 5. Produce a **ranked list** of roommates (best → worst)
Include 1–2 sentences explaining the ranking.

### 6. Choose ONE best apartment from the RAG results
Justify the apartment using:
- neighborhood
- price
- quiet vs social environment
- amenities (private bathroom, pets allowed)
- sunrise/sunset exposure if relevant
- any detail explicitly present in the RAG context

### 7. Final output format (MANDATORY):

#### 🧑‍🤝‍🧑 Best Roommate Match
(name or label)

#### 🔢 Compatibility Score: XX/100

#### 📊 Compatibility Matrix
(list the 10 scores)

#### ⚠️ Conflict Risks
- Risk (severity)

#### 🏠 Best Apartment Match
- Apartment title
- Why it fits both the user AND the chosen roommate

#### 🟣 Final Verdict (≤ 4 sentences)
Give a clear recommendation.

STRICT RULE:
You MUST NOT hallucinate details not present in the RAG context.
If a detail is unknown, say “not specified”.

Begin with the multi-candidate analysis now.
"""

    return run_llm(prompt)

# --- 5. Quick manual test ---
if __name__ == "__main__":
    q = "I want a quiet big dog friendly apartment in barrio salamanca with a budget of 3300 euros a month a max, i want my own private bathroom and sunset views"
    print(recommend(q))
