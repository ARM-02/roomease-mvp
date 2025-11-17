from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, requests, json, re

import chromadb
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
from google.generativeai.types import GenerationConfig


# ------------------------------
#   GEMINI CONFIG
# ------------------------------


YOUR_API_KEY = "AIzaSyAwxwUXvx-PuxRr1iVEGiF3hSpL-W--_rY"  # <-- CHANGED

if YOUR_API_KEY == "YOUR_API_KEY_HERE":
    print("="*50)
    print("ERROR: Please update 'YOUR_API_KEY' in the script.")
    print("="*50)
    exit()

# Configure the API with the variable
genai.configure(api_key=YOUR_API_KEY)  # <-- CHANGED

# 3. Use a correct, powerful model name
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"  # <-- CHANGED


def run_gemini(prompt: str, json_schema: Optional[Dict] = None) -> str:
    """Send prompt to Gemini via the API safely."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        gen_config = GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048
        )

        if json_schema:
            gen_config.response_mime_type = "application/json"
            gen_config.response_schema = json_schema

        response = model.generate_content(
            prompt,
            generation_config=gen_config
        )

        # --- FIX: handle missing parts safely ---
        if not response.candidates:
            return "ERROR: Gemini returned no candidates."

        candidate = response.candidates[0]

        if not candidate.content.parts:
            return f"ERROR: Gemini returned an empty completion. finish_reason={candidate.finish_reason}"

        # If JSON schema was applied, response.text is safe:
        if json_schema:
            return response.text

        # Otherwise, extract text manually
        out = ""
        for part in candidate.content.parts:
            if hasattr(part, "text"):
                out += part.text

        return out if out.strip() else "(empty response)"

    except Exception as e:
        return f"ERROR: Gemini API failed: {e}"


# ------------------------------
#   1. Load Sentence Embeddings
# ------------------------------
print("[INFO] Loading embedding model...")
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)


def embed_query(text: str) -> List[List[float]]:
    return embedder.encode([text], normalize_embeddings=True).tolist()


# ------------------------------
#   2. Connect to ChromaDB
# ------------------------------
print("[INFO] Connecting to ChromaDB...")

DB_PATH = "./chroma_store"
APTS_COLLECTION = "apartments"
STUDENTS_COLLECTION = "students"

client = chromadb.PersistentClient(path=DB_PATH)

try:
    apts = client.get_collection(APTS_COLLECTION)
    students = client.get_collection(STUDENTS_COLLECTION)
    print("[INFO] Collections loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load collections: {e}")
    exit()


# ------------------------------
#   JSON PARSER FOR LLM OUTPUT
# ------------------------------
def _safe_json_parse(llm_response: str) -> Dict:
    # This is still a good fallback, but the new run_gemini makes it more robust
    try:
        match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if not match:
            print(f"[WARN] No JSON object found in LLM response: {llm_response}")
            return {}
        return json.loads(match.group(0))
    except Exception as e:
        print(f"[WARN] JSON parse failed: {e}")
        return {}


# ------------------------------
#   Format Helpers
# ------------------------------
def _format_apartment_context(res: Dict[str, Any]) -> str:
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    lines = []
    for doc, meta in zip(docs, metas):
        prop_id = meta.get("propertyCode")
        hood = meta.get("neighborhood") or meta.get("district") or "Unknown"
        price = meta.get("price")
        url = meta.get("url")

        # parse title inside suggestedTexts
        title = f"Property {prop_id}"
        stxt = meta.get("suggestedTexts")
        if stxt:
            # Use regex to find title, safer than eval()
            match = re.search(r"'title':\s*'([^']+)'", stxt)
            if match:
                title = match.group(1)

        header = (
            f"- PROPERTY_CODE: {prop_id} | {title} | "
            f"Neighborhood: {hood} | Price: €{price} | URL: {url}"
        )
        lines.append(f"{header}\n  Description: {doc}")

    return "\n\n".join(lines)


def _format_student_context(res):
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    lines = []
    for doc, meta in zip(docs, metas):
        lines.append(
            f"- Candidate — {meta.get('name')}\n"
            f"  Personality: {meta.get('personality')}\n"
            f"  Lifestyle: {meta.get('lifestyle_summary')}\n"
            f"  Sleep schedule: {meta.get('sleep_schedule')}\n"
            f"  Noise tolerance: {meta.get('noise_tolerance')}\n"
            f"  Dog friendliness: {meta.get('dog_friendliness')}\n"
            f"  Cleanliness: {meta.get('cleanliness')}\n"
            f"  Study habits: {meta.get('study_habits')}\n"
            f"  Raw text: {doc[:200]}..."
        )

    return "\n\n".join(lines)


# ------------------------------
#   4. MAIN RECOMMEND() FUNCTION
# ------------------------------
def recommend(query: str, n_apts: int = 3, n_students: int = 6) -> str:
    # Step 0: Disambiguate user request
    print("[INFO] Step 0: Disambiguating query with Gemini...")

    # --- NEW: Define the *exact* JSON schema we want ---
    disambiguation_schema = {
        "type": "OBJECT",
        "properties": {
            "apartment_query": {"type": "STRING"},
            "user_profile": {"type": "STRING"}
        },
        "required": ["apartment_query", "user_profile"]
    }

    disambiguation_prompt = f"""
Separate the user's request into two fields:
1. apartment_query  → filters for the apartment (location, price, vibe)
2. user_profile     → personal traits for roommate matching (personality, lifestyle)

User request:
"{query}"
"""

    # --- Call Gemini and force JSON output ---
    llm_response = run_gemini(
        disambiguation_prompt,
        json_schema=disambiguation_schema  # <-- CHANGED
    )

    # Now, llm_response *is* the JSON string, no need for regex
    queries = json.loads(llm_response)  # <-- CHANGED (removed _safe_json_parse)

    apartment_query = queries.get("apartment_query", query)
    user_profile = queries.get("user_profile", query)

    print("[INFO] Apartment Query:", apartment_query)
    print("[INFO] User Profile:", user_profile)

    # Step 1: Retrieve results from Chroma
    print("[INFO] Retrieving from ChromaDB...")

    try:
        apt_res = apts.query(
            query_embeddings=embed_query(apartment_query),
            n_results=n_apts
        )
        stu_res = students.query(
            query_embeddings=embed_query(user_profile),
            n_results=n_students
        )
    except Exception as e:
        return f"ERROR: Chroma query failed: {e}"

    apt_context = _format_apartment_context(apt_res)
    stu_context = _format_student_context(stu_res)

    # Step 2: Final synthesis using Gemini
    print("[INFO] Step 2: Synthesizing with Gemini...")

    final_prompt = f"""
USER PROFILE:
\"\"\"{user_profile}\"\"\"

---
APARTMENTS (retrieved):
{apt_context}

---
CANDIDATES (retrieved):
{stu_context}

---
TASK:
Summarize the user's key traits.
Then evaluate EACH candidate comparing their personality/routines/dog-compatibility directly with the user.

Finally:
Return the **Best Roommate** and **Best Apartment** exactly as:

**Best Roommate:** <name>
**Best Apartment PROPERTY_CODE:** <code>
**Best Apartment URL:** <url>
**Recommendation:** <1–2 sentences>
"""

    # This call is for simple text, so no JSON schema is needed
    return run_gemini(final_prompt)  # <-- CHANGED


# Manual test
if __name__ == "__main__":

        q = (
            "I want a quiet big dog friendly apartment in barrio Salamanca with a budget of "
            "3300 euros max. I want my own private bathroom and sunset views. "
            "I am very extroverted, not very organised and spontaneous, and I cannot handle "
            "people that are too structured."
        )

        print("--- Running Recommendation Engine ---")
        result = recommend(q)
        print("\n--- FINAL RECOMMENDATION ---")
        print(result)