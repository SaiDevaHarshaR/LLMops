#pip install requests faiss-cpu sentence-transformers
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Embedding model (to turn text into vectors)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Knowledge base (mini docs)
docs = [
    "Password resets are done from the account settings page.",
    "Contact support for multi-factor authentication issues.",
    "Docker helps package and run applications."
]

# 3. Embed docs and store in FAISS
doc_embeddings = model.encode(docs)
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings))

# 4. Query
query = "How do I reset my password?"
query_vec = model.encode([query])
distances, indices = index.search(np.array(query_vec), k=2)

# 5. Build context
context = "\n".join(docs[i] for i in indices[0])

# 6. Ask local LLM via Ollama API
prompt = f"Use ONLY this context:\n{context}\n\nQuestion: {query}\nAnswer briefly:"

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.2", "prompt": prompt, "stream": False}
)

print("Query:", query)
print("Answer:", response.json()["response"])
