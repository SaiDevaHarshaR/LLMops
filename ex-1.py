# Install dependencies before running:
# pip install faiss-cpu sentence-transformers

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Documents (mini knowledge base)
docs = [
    "AWS has many services for cloud computing.",
    "Python is used in AI and data science.",
    "Docker helps package and run applications."
]

# 3. Convert documents to vector embeddings
doc_embeddings = model.encode(docs)

# 4. Create FAISS index
dimension = doc_embeddings.shape[1]  # size of each vector
index = faiss.IndexFlatL2(dimension) # L2 = Euclidean distance

# 5. Add docs to FAISS index
index.add(np.array(doc_embeddings))

# 6. Create a query
query = "What is used in AI?"
query_embedding = model.encode([query])

# 7. Search the index (k=1 means return top 1 match)
distances, indices = index.search(np.array(query_embedding), k=1)

# 8. Print result
print("Query:", query)
print("Most similar doc:", docs[indices[0][0]])
