import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# load vector DB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("apartments")

# load LLM
model_id = "microsoft/phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
llm = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=250)


def recommend(query):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    context = "\n\n".join(results["documents"][0])
    prompt = f"""
You are a housing assistant. 
User request: {query}

Apartment options:
{context}

Which apartment best fits the user and why?
"""

    output = llm(prompt)[0]["generated_text"]
    return output


if __name__ == "__main__":
    print(recommend("I want a quiet place to study near IE University"))
