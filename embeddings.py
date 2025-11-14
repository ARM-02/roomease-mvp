from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd

def build_chroma_db():
    df = pd.read_csv("data/apartments.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create persistent DB folder "chroma_db" in project root
    client = chromadb.PersistentClient(path="chroma_db")

    # Create collection
    collection = client.create_collection("apartments")

    # Generate embeddings
    embeddings = model.encode(df["description"].tolist()).tolist()

    # Store in Chroma
    collection.add(
        documents=df["description"].tolist(),
        metadatas=df.to_dict(orient="records"),
        ids=[str(i) for i in df["id"]],
        embeddings=embeddings
    )

    print("✅ ChromaDB built with", len(df), "records.")
    return collection

if __name__ == "__main__":
    build_chroma_db()
