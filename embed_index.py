from __future__ import annotations
import os, re, uuid, argparse, sys
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ======================= Config =======================
DB_PATH = "./chroma_store"
APTS_COLLECTION = "apartments"
EMBED_MODEL = "all-MiniLM-L6-v2"     # fast, local, great default

# ======================= Utils ========================
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

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
    title_col = next((c for c in df.columns if c.lower() in {"title", "name"}), None)
    desc_col  = next((c for c in df.columns if c.lower() in {"description", "desc"}), None)
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

# =================== Chroma helpers ====================
def get_collection(client: PersistentClient, name: str):
    try:
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    except TypeError:
        return client.get_or_create_collection(name=name)

def add_rows(collection, embedder: LocalEmbedder, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    docs  = [r["document"] for r in rows]
    ids   = [r["id"] for r in rows]
    metas = [r["metadata"] for r in rows]
    embs  = embedder.embed(docs)
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    return len(rows)

def quick_query(collection, embedder: LocalEmbedder, text: str, k: int = 5):
    qemb = embedder.embed([text])[0]
    res = collection.query(
        query_embeddings=[qemb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return res

# ========================= Main ========================
def main():
    ap = argparse.ArgumentParser(description="Embed apartments CSV into Chroma.")
    ap.add_argument("--csv", default="data/available_apartments.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--reset-apartments", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[error] CSV not found: {args.csv}")

    os.makedirs(args.db, exist_ok=True)

    # 1) Load apartment rows
    apt_rows = rows_from_apartments_csv(args.csv)
    print(f"[load] apartments rows={len(apt_rows)}")

    # 2) Init embedder + Chroma
    embedder = LocalEmbedder(args.model)
    client = PersistentClient(path=args.db)

    if args.reset_apartments:
        try:
            client.delete_collection(APTS_COLLECTION)
            print(f"[reset] deleted collection '{APTS_COLLECTION}'")
        except Exception:
            pass

    apts = get_collection(client, APTS_COLLECTION)

    # 3) Index
    n1 = add_rows(apts, embedder, apt_rows)
    print(f"[index] apartments +{n1}")
    print(f"[persist] path={os.path.abspath(args.db)} | collections=({APTS_COLLECTION})")

    # 4) Optional demo
    if args.demo:
        print("\n[demo] apartments:")
        res = quick_query(apts, embedder, "quiet, bright room in Salamanca under 1000 euros", 5)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hood = meta.get("neighborhood", meta.get("Neighborhood", "?"))
            price = meta.get("price", meta.get("Price", "?"))
            print(f"  dist={dist:.4f} | {hood} | €{price}")
            print("   ", doc[:200], ("..." if len(doc) > 200 else ""))

if __name__ == "__main__":
    main()
