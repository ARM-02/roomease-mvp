from __future__ import annotations
from typing import List, Dict, Any
import os, requests, json, re

import chromadb
from sentence_transformers import SentenceTransformer

# --- Config: MUST match your index scripts ---
DB_PATH = "./chroma_store"
APTS_COLLECTION = "apartments"
STUDENTS_COLLECTION = "students"
EMBED_MODEL = "all-MiniLM-L6-v2"

# LM Studio HTTP server
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

# --- 1. Load embedder (same model as embeddings) ---
print("[INFO] Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)


def embed_query(text: str) -> List[List[float]]:
    """Encodes a single query string."""
    return embedder.encode([text], normalize_embeddings=True).tolist()


# --- 2. Load Chroma persistent DB ---
print("[INFO] Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=DB_PATH)
try:
    apts = client.get_collection(APTS_COLLECTION)
    students = client.get_collection(STUDENTS_COLLECTION)
    print("[INFO] Collections loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load collections: {e}")
    print("Please make sure you have run the indexing script first.")
    exit()


# --- 3. LM Studio client ---
def run_llm(prompt: str) -> str:
    """Send prompt to local LM Studio model."""
    try:
        resp = requests.post(
            LMSTUDIO_URL,
            json={
                "model": "deepseek-r1-distill-llama-8b",  # must match the model name
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Failed to connect to LM Studio at {LMSTUDIO_URL}")
        print("Please ensure LM Studio is running and the server is on.")
        return f"ERROR: LLM connection failed: {e}"


def _safe_json_parse(llm_response: str) -> Dict:
    """
    Finds and parses the first valid JSON object in a (potentially messy) LLM string.
    """
    try:
        match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if not match:
            print(f"[WARN] No JSON object found in LLM response: {llm_response}")
            return {}
        json_string = match.group(0)
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"[WARN] Failed to decode JSON: {e}")
        return {}
    except Exception as e:
        print(f"[WARN] Unknown error during JSON parse: {e}")
        return {}


# --- Format helpers (UPDATED) ---
def _format_apartment_context(res: Dict[str, Any]) -> str:
    """Pretty-print top apartment hits using correct CSV columns."""
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    lines = []
    for doc, meta in zip(docs, metas):
        # Use the *exact* column names from available_apartments.csv
        prop_id = meta.get("propertyCode") or f"UNKNOWN_ID_{meta.get('chunk_index', 0)}"
        hood = meta.get("neighborhood") or meta.get("district") or "Unknown Area"
        price = meta.get("price") or "Price N/A"
        url = meta.get("url") or "No URL"

        # Try to get title from the 'suggestedTexts' field
        # We must parse it, as it's a string representation of a dict
        title = f"Property {prop_id}"  # Fallback title
        try:
            # suggestedTexts might be a string: "{'title': 'Piso...'}"
            suggested_texts_str = meta.get("suggestedTexts")
            if suggested_texts_str:
                # Use regex to find title, safer than eval()
                title_match = re.search(r"'title':\s*'([^']+)'", suggested_texts_str)
                if title_match:
                    title = title_match.group(1)
        except Exception:
            pass  # Keep the fallback title

        header_parts = [
            f"PROPERTY_CODE: {prop_id}",
            title,
            f"Neighborhood: {hood}",
            f"Price: €{price}",
            f"URL: {url}"
        ]

        header = " | ".join(header_parts)
        lines.append(f"- {header}\n  Description: {doc}")

    return "\n\n".join(lines) if lines else "No apartments found in the database."


def _format_student_context(res):
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    lines = []
    for doc, meta in zip(docs, metas):
        # These keys come from your 'extract_students_metadata.py' script
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
    Runs the full RAG pipeline.
    """

    # --- Step 0: Disambiguate the Query ---
    print("[INFO] Step 0: Disambiguating query...")
    disambiguation_prompt = f"""
Read the user's request. Separate it into two parts:
1. The 'apartment_query': The user's preferences for the apartment itself (location, size, price, vibe).
2. The 'user_profile': The user's description of their own personality and lifestyle (extroverted, messy, etc.).

Return ONLY a valid JSON object like this:
{{"apartment_query": "...", "user_profile": "..."}}

User Request:
"{query}"
"""
    llm_response = run_llm(disambiguation_prompt)
    if "ERROR" in llm_response: return llm_response  # Pass on error

    queries = _safe_json_parse(llm_response)

    apartment_query = queries.get("apartment_query", query)
    user_profile = queries.get("user_profile", query)

    print(f"[INFO] Apartment Query: {apartment_query}")
    print(f"[INFO] User Profile: {user_profile}")

    # --- Step 1: Retrieve data from Chroma ---
    print("[INFO] Step 1: Retrieving data from ChromaDB...")
    try:
        apt_emb = embed_query(apartment_query)
        stu_emb = embed_query(user_profile)

        apt_res = apts.query(
            query_embeddings=apt_emb,
            n_results=n_apts,
        )

        stu_res = students.query(
            query_embeddings=stu_emb,
            n_results=n_students,
        )
    except Exception as e:
        print(f"[ERROR] ChromaDB query failed: {e}")
        return f"ERROR: ChromaDB query failed: {e}"

    apt_context = _format_apartment_context(apt_res)
    stu_context = _format_student_context(stu_res)

    # --- Step 2: LLM Synthesis (UPDATED PROMPT) ---
    print("[INFO] Step 2: Sending data to LLM for final synthesis...")
    prompt = f"""
You are an expert roommate-matching AI. Your task is to find the best roommate and apartment for the user.

USER PROFILE:
\"\"\"{user_profile}\"\"\"

---
APARTMENTS AVAILABLE (Context):
{apt_context}

---
POTENTIAL ROOMMATE CANDIDATES (Context):
{stu_context}

---
TASK:
First, summarize the user's key traits in 2 bullet points.
Second, analyze each candidate one-by-one by *directly comparing* them to the user.

**User Key Traits:**
* **Social:** [Summarize the user's social habits]
* **Lifestyle:** [Summarize the user's cleanliness and daily routine]

---
**Candidate Analysis:**

**1. Candidate: [Name]**
* **Social Match:** [Compare the user's social trait to this candidate's. Is it a good or bad match?]
* **Lifestyle Match:** [Compare the user's lifestyle/cleanliness trait to this candidate's. Is it a good or bad match?]
* **Verdict:** [State "Good Match", "Neutral Match", or "Bad Match"]
* **Reasoning:** [1-2 sentences explaining your verdict.]

**2. Candidate: [Name]**
* **Social Match:** [Compare...]
* **Lifestyle Match:** [Compare...]
* **Verdict:** [State...]
* **Reasoning:** [Explain...]

(Repeat this analysis for ALL candidates)

---
**Final Recommendation:**
Based on your side-by-side analysis, identify the #1 Best Roommate and the #1 Best Apartment.

You MUST output the apartment's `PROPERTY_CODE` and `URL` exactly as they were given.

Format your answer like this:

**Best Roommate:** [Candidate Name]
**Best Apartment PROPERTY_CODE:** [The exact PROPERTY_CODE from the context]
**Best Apartment URL:** [The exact URL from the context]
**Recommendation:** [1-2 sentences justifying why this pair and apartment are a good fit.]
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

    print("--- Running Recommendation Engine ---")
    recommendation = recommend(q)
    print("\n--- FINAL RECOMMENDATION ---")
    print(recommendation)