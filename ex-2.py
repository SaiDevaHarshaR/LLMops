import os
import pinecone
from sentence_transformers import SentenceTransformer

# 1. Initialize Pinecone
pinecone.init(api_key=os.getenv("pcsk_5gZp8g_AEDepRJENz3ns14izzf9YETHvdPpfa1CcafcuTDJujGdsS4mHsnCJYXmRLVieK5"), environment="us-west1-gcp-free")  # free starter env

# 2. Create an index (if not already created)
index_name = "demo-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384)  # 384 matches our embedding size

index = pinecone.Index(index_name)

# 3. Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Documents and their embeddings
docs = [
    "AWS offers many cloud services.",
    "Python is popular for AI and ML.",
    "Docker containers package applications."
]
vectors = model.encode(docs).tolist()

# 5. Upload them to Pinecone (upsert)
index.upsert(vectors=list(zip(map(str, range(len(vectors))), vectors)))

# 6. Query vector
query = "Which is used for AI?"
query_vec = model.encode([query]).tolist()[0]
result = index.query(query_vec, top_k=1, include_metadata=False)

print("Most similar doc:", docs[int(result['matches'][0]['id'])])
