from __future__ import annotations
import os, re, sys, json, argparse
from typing import List, Dict

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import requests

DB_PATH = "./chroma_store"
STUDENTS_COLLECTION = "students"
EMBED_MODEL = "all-MiniLM-L6-v2"

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions" # LM Studio HTTP server

# ======================= Utils =========================
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, str]:
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = "not specified"
        else:
            clean[k] = str(v)
    return clean

def split_students(raw_text: str) -> List[str]:
    """
    Heuristic split of PDF text into per-student blocks.
    Adjust the regex if your PDF has a different structure.
    """
    # Example: names like "Briana Thomas", "Lisa Williamson" on their own line
    parts = re.split(r"(?=STUDENT \d+)", raw_text)
    if len(parts) <= 2:
        # Fallback: big chunks if the regex fails
        approx_size = 1200
        return [
            raw_text[i:i + approx_size]
            for i in range(0, len(raw_text), approx_size)
        ]
    return [p.strip() for p in parts if p.strip()]

def ask_local_llm(prompt: str) -> Dict:
    """Call LM Studio local model and parse JSON."""
    resp = requests.post(
        LMSTUDIO_URL,
        json={
            "model": "llama-3.2-1b-instruct",   # in LM Studio: set this to your loaded model name
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 400,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Find the first JSON block in the (possibly messy) LLM response
    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in LLM response")

    json_string = match.group(0)
    return json.loads(json_string)

EXTRACTION_PROMPT = """
You will read one student's personality / housing preference response
and extract structured metadata.

Return ONLY valid JSON with the following fields:

{
  "name": string | null,
  "personality": string,
  "lifestyle_summary": string,
  "sleep_schedule": string,
  "noise_tolerance": string,
  "dog_friendliness": string,
  "cleanliness": string,
  "study_habits": string
}

Rules:
- If a field is unknown, write "not specified".
- Do NOT add extra fields.
- Do NOT comment.
- No markdown.
Text:
"""


def extract_student_metadata(block: str) -> Dict:
    # 🎯 REFINED HEURISTIC: Find the name following "STUDENT X" or "STUDENT X -"

    # Pattern looks for:
    # 1. "STUDENT " followed by 1 or more digits (X)
    # 2. OPTIONAL: whitespace and a hyphen (to catch "STUDENT 41 - Camila Moreira")
    # 3. CAPTURE GROUP 1: The name, starting with an uppercase letter, followed by
    #    at least one other word (First Name Last Name)
    name_match = re.search(r"STUDENT\s+\d+\s*\-*\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", block)

    # Clean up the captured name (sometimes includes surrounding spaces or noise)
    fallback_name = normalize_space(name_match.group(1)) if name_match else None

    prompt = EXTRACTION_PROMPT + "\n" + block
    try:
        meta = ask_local_llm(prompt)

        # FIX: If the LLM returned "not specified" for the name, use the fallback
        if meta.get("name") in [None, "not specified", ""]:
            meta["name"] = fallback_name

        return meta

    except Exception as e:
        print(f"[WARN] Extraction failed for one block: {e}")
        # Fallback: neutral defaults, but include the heuristically extracted name
        return {
            "name": fallback_name,  # Use the heuristically extracted name here
            "personality": "not specified",
            "lifestyle_summary": "not specified",
            "sleep_schedule": "not specified",
            "noise_tolerance": "not specified",
            "dog_friendliness": "not specified",
            "cleanliness": "not specified",
            "study_habits": "not specified",
        }

def embed_texts(model, texts: List[str]):
    return model.encode(texts, normalize_embeddings=True).tolist()

# =========================== Main ======================
def main():
    ap = argparse.ArgumentParser(description="Index students from PDF into Chroma with metadata using LM Studio.")
    ap.add_argument("--pdf", required=True, help="data/fake_student_ocean_responses_long")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--reset", action="store_true", help="Delete students collection before indexing")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        sys.exit(f"[ERROR] PDF not found: {args.pdf}")

    # Load PDF
    print("[INFO] Loading PDF…")
    reader = PdfReader(args.pdf)
    raw = "\n".join([page.extract_text() or "" for page in reader.pages])
    raw = normalize_space(raw)

    print("[INFO] Splitting into student entries…")
    students_blocks = split_students(raw)
    print(f"[INFO] Found ~{len(students_blocks)} possible students.")

    # Connect to Chroma
    os.makedirs(args.db, exist_ok=True)
    client = PersistentClient(path=args.db)

    if args.reset:
        try:
            client.delete_collection(STUDENTS_COLLECTION)
            print(f"[INFO] Reset collection '{STUDENTS_COLLECTION}'")
        except Exception:
            pass

    col = client.get_or_create_collection(
        STUDENTS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    embedder = SentenceTransformer(args.model)

    for i, block in enumerate(students_blocks):
        print(f"[INFO] Extracting + embedding student {i+1}/{len(students_blocks)}…")
        meta = extract_student_metadata(block)
        # enrich meta with technical fields
        meta["source"] = os.path.abspath(args.pdf)
        meta["chunk_index"] = i

        # sanitize before saving
        meta = sanitize_metadata(meta)

        emb = embed_texts(embedder, [block])[0]

        col.add(
            ids=[f"student::{i}"],
            embeddings=[emb],
            documents=[block],
            metadatas=[meta],
        )

    print("[INFO] DONE. Students collection updated.")
    print(f"[INFO] Path: {os.path.abspath(args.db)}")

if __name__ == "__main__":
    main()

