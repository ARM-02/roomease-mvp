from __future__ import annotations
import os, re, uuid, argparse, sys
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ======================= Config =======================
DB_PATH = "./chroma_store"
APTS_COLLECTION = "apartments"
STUDENTS_COLLECTION = "students"
EMBED_MODEL = "all-MiniLM-L6-v2"     # fast, local, great default
PDF_CHUNK_WORDS = 80                  # ≈ 80 words per chunk

# ======================= Utils ========================
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def chunk_text(text: str, words_per_chunk: int) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i:i+words_per_chunk]).strip()
        for i in range(0, len(words), words_per_chunk)
        if words[i:i+words_per_chunk]
    ]

# =================== Embedding model ===================
class LocalEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return L2-normalized embeddings for cosine distance in Chroma."""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

# ================= CSV → apartment rows ================
def rows_from_apartments_csv(csv_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        sys.exit(f"[error] CSV not found: {csv_path}")
    df = pd.read_csv(csv_path).fillna("")
    # Try common columns (optional)
    title_col = next((c for c in df.columns if c.lower() in {"title","name"}), None)
    desc_col  = next((c for c in df.columns if c.lower() in {"description","desc"}), None)
    hood_col  = next((c for c in df.columns if "neigh" in c.lower()), None)
    price_col = next((c for c in df.columns if "price" in c.lower()), None)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        title = normalize_space(row.get(title_col, "")) if title_col else ""
        desc  = normalize_space(row.get(desc_col, "")) if desc_col else ""
        hood  = normalize_space(row.get(hood_col, "")) if hood_col else ""
        price = str(row.get(price_col, "")) if price_col else ""

        parts = []
        if title: parts.append(title)
        if desc:  parts.append(desc)
        if hood:  parts.append(f"Neighborhood: {hood}")
        if price: parts.append(f"Price: €{price}")
        doc = normalize_space(". ".join(parts)) or normalize_space(str(row.to_dict()))

        rows.append({
            "id": f"apt::{uuid.uuid4()}",
            "document": doc,
            "metadata": {**row.to_dict(), "type": "apartment"}
        })
    return rows

# ============== PDF → student chunk rows ===============
def rows_from_students_pdf(pdf_path: str, words_per_chunk: int) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        sys.exit(f"[error] PDF not found: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    text = normalize_space("\n".join(pages))
    chunks = chunk_text(text, words_per_chunk)

    rows: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        rows.append({
            "id": f"studentpdf::{os.path.basename(pdf_path)}::{i}",
            "document": ch,
            "metadata": {
                "type": "student_response",
                "source": os.path.abspath(pdf_path),
                "chunk_index": i,
                "chunk_words": words_per_chunk
            }
        })
    return rows

# =================== Chroma helpers ====================
def get_collection(client: PersistentClient, name: str):
    # Prefer cosine for normalized sentence-transformer embeddings
    try:
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    except TypeError:
        return client.get_or_create_collection(name=name)

def add_rows(collection, embedder: LocalEmbedder, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    docs = [r["document"] for r in rows]
    ids  = [r["id"] for r in rows]
    metas= [r["metadata"] for r in rows]
    embs = embedder.embed(docs)
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    return len(rows)

# ================ Minimal query (optional) =============
def quick_query(collection, embedder: LocalEmbedder, text: str, k: int = 5):
    qemb = embedder.embed([text])[0]
    res = collection.query(query_embeddings=[qemb], n_results=k,
                           include=["documents","metadatas","distances"])
    return res

# ========================= Main ========================
def main():
    ap = argparse.ArgumentParser(description="Embed CSV (apartments) + PDF (students) into Chroma.")
    ap.add_argument("--csv", default="available_apartments.csv")
    # updated default to v2; if missing, we'll try the old filename as a fallback
    ap.add_argument("--pdf", default="fake_student_ocean_responses_long_v2.pdf")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--chunk_words", type=int, default=PDF_CHUNK_WORDS)
    ap.add_argument("--demo", action="store_true", help="Run a tiny demo query after indexing")
    ap.add_argument("--reset-students", action="store_true", help="Delete the 'students' collection before reindexing")
    args = ap.parse_args()

    # Resolve PDF path (fallback to old name if user hasn't updated)
    if not os.path.exists(args.pdf):
        fallback = "fake_student_ocean_responses_long.pdf"
        if os.path.exists(fallback):
            print(f"[warn] PDF not found: {args.pdf} → using fallback: {fallback}")
            args.pdf = fallback
        else:
            sys.exit(f"[error] PDF not found: {args.pdf}")

    if not os.path.exists(args.csv):
        sys.exit(f"[error] CSV not found: {args.csv}")

    os.makedirs(args.db, exist_ok=True)

    # 1) Load data
    apt_rows = rows_from_apartments_csv(args.csv)
    stu_rows = rows_from_students_pdf(args.pdf, args.chunk_words)
    print(f"[load] apartments rows={len(apt_rows)} | student chunks={len(stu_rows)}")

    # 2) Init embedder + Chroma
    embedder = LocalEmbedder(args.model)
    client = PersistentClient(path=args.db)

    # Optional: reset students collection to avoid mixing old/new PDFs
    if args.reset_students:
        try:
            client.delete_collection(STUDENTS_COLLECTION)
            print(f"[reset] deleted collection '{STUDENTS_COLLECTION}'")
        except Exception:
            pass

    apts = get_collection(client, APTS_COLLECTION)
    students = get_collection(client, STUDENTS_COLLECTION)

    # 3) Index
    n1 = add_rows(apts, embedder, apt_rows)
    n2 = add_rows(students, embedder, stu_rows)
    print(f"[index] apartments +{n1} | students +{n2}")
    print(f"[persist] path={os.path.abspath(args.db)} | collections=({APTS_COLLECTION}, {STUDENTS_COLLECTION})")

    # 4) Optional: quick demo (safe to remove)
    if args.demo:
        print("\n[demo] apartments:")
        res = quick_query(apts, embedder, "quiet, bright room in Salamanca under 1000 euros", 5)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            print(f"  dist={dist:.4f} | {meta.get('neighborhood', meta.get('Neighborhood','?'))} | €{meta.get('price', meta.get('Price','?'))}")
            print("   ", doc[:200], ("..." if len(doc) > 200 else ""))
        print("\n[demo] students:")
        res2 = quick_query(students, embedder, "calm apartment, organized, early sleeper", 3)
        for doc, meta, dist in zip(res2["documents"][0], res2["metadatas"][0], res2["distances"][0]):
            print(f"  dist={dist:.4f} | chunk={meta.get('chunk_index')}")
            print("   ", doc[:200], ("..." if len(doc) > 200 else ""))

if __name__ == "__main__":
    main()

