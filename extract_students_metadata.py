from __future__ import annotations
import os, re, sys, uuid
from typing import List, Dict
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

DB_PATH = "./chroma_store"
STUDENTS_COLLECTION = "students"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_CHUNK_WORDS = 80

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def chunk_text(text: str, words_per_chunk: int = PDF_CHUNK_WORDS) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i:i+words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
        if words[i:i+words_per_chunk]
    ]

def load_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        sys.exit(f"[ERROR] PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return normalize_space("\n".join(pages))

def embed_texts(model, texts: List[str]):
    return model.encode(texts, normalize_embeddings=True).tolist()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to the student PDF")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    # Load data
    print("[INFO] Loading PDF…")
    raw_text = load_pdf(args.pdf)
    chunks = chunk_text(raw_text)
    print(f"[INFO] PDF loaded — {len(chunks)} chunks")

    # Chroma setup
    os.makedirs(args.db, exist_ok=True)
    client = PersistentClient(path=args.db)

    if args.reset:
        try:
            client.delete_collection(STUDENTS_COLLECTION)
            print("[INFO] Reset students collection")
        except Exception:
            pass

    students_col = client.get_or_create_collection(
        name=STUDENTS_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # Embed
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_texts(model, chunks)

    ids = [f"chunk::{i}" for i in range(len(chunks))]
    metas = [{"chunk_index": i, "source": args.pdf} for i in range(len(chunks))]

    print("[INFO] Adding to Chroma…")
    students_col.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metas,
    )

    print("[INFO] Done. Students collection updated.")
    print(f"[INFO] Path: {os.path.abspath(args.db)}")

if __name__ == "__main__":
    main()
