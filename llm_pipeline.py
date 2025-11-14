from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import chromadb
from sentence_transformers import SentenceTransformer

def get_recommendations(query):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.get_collection("apartments")

    query_embedding = embedder.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=2)

    # Combine retrieved text for context
    context = "\n".join(results["documents"][0])
    prompt = f"Student request: {query}\n\nOptions:\n{context}\n\nWhich one fits best and why?"

    model_id = "microsoft/phi-3-mini-4k-instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=200)

    output = gen(prompt)[0]["generated_text"]
    print("✅ Recommendation:\n", output)

if __name__ == "__main__":
    get_recommendations("quiet apartment near IE for under 1000 euros")
