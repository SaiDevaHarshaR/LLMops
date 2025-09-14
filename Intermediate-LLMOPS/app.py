#install all the dependencies `pip install fastapi uvicorn llama-index llama-index-llms-ollama llama-index-embeddings-huggingface pypdf`

from fastapi import FastAPI, Query
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate

# -----------------------------
# 1. FastAPI app
# -----------------------------
app = FastAPI(title="RAG API with LlamaIndex + Weaviate + Ollama")

# -----------------------------
# 2. Configure embeddings + LLM
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3.2")

# -----------------------------
# 3. Connect to Weaviate
# -----------------------------
client = weaviate.Client("http://weaviate:8080")

# -----------------------------
# 4. Load PDF documents
# -----------------------------
documents = SimpleDirectoryReader("./data").load_data()

# -----------------------------
# 5. Store documents in Weaviate
# -----------------------------
vector_store = WeaviateVectorStore(weaviate_client=client, index_name="DocsIndex")

# Build index connected to Weaviate
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
query_engine = index.as_query_engine()

# -----------------------------
# 6. Define API endpoint
# -----------------------------
@app.get("/ask")
def ask_question(q: str = Query(..., description="Your question")):
    response = query_engine.query(q)
    return {"question": q, "answer": str(response)}







"""
let's start with logging which is required to check the logs for LLM calls..
install `pip install prometheus-client` using CLI


import time
import logging
from fastapi import FastAPI, Query, Request
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_app.log"),  # save to file
        logging.StreamHandler()              # print to console
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="RAG API with LlamaIndex + Weaviate + Ollama")

# -----------------------------
# Configure embeddings + LLM
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3.2")

# -----------------------------
# Connect to Weaviate
# -----------------------------
client = weaviate.Client("http://weaviate:8080")

# -----------------------------
# Load documents
# -----------------------------
documents = SimpleDirectoryReader("./data").load_data()
vector_store = WeaviateVectorStore(weaviate_client=client, index_name="DocsIndex")
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
query_engine = index.as_query_engine()

# -----------------------------
# API endpoint with logging
# -----------------------------
@app.get("/ask")
async def ask_question(q: str = Query(..., description="Your question"), request: Request = None):
    start_time = time.time()

    response = query_engine.query(q)
    elapsed = round(time.time() - start_time, 2)

    # log details
    client_ip = request.client.host if request else "unknown"
    logger.info(f"Client {client_ip} asked: {q} | Answer: {response} | Time: {elapsed}s")

    return {"question": q, "answer": str(response), "latency_sec": elapsed}

    


let's add metrics to this python.py
from prometheus_client import Counter, Histogram
from fastapi.responses import PlainTextResponse

# -----------------------------
# Metrics setup
# -----------------------------
REQUEST_COUNT = Counter("rag_requests_total", "Total number of RAG requests")
REQUEST_LATENCY = Histogram("rag_request_latency_seconds", "Latency of RAG requests")

@app.get("/ask")
@REQUEST_LATENCY.time()
async def ask_question(q: str = Query(...), request: Request = None):
    REQUEST_COUNT.inc()  # increment request count
    start_time = time.time()

    response = query_engine.query(q)
    elapsed = round(time.time() - start_time, 2)

    client_ip = request.client.host if request else "unknown"
    logger.info(f"Client {client_ip} asked: {q} | Answer: {response} | Time: {elapsed}s")

    return {"question": q, "answer": str(response), "latency_sec": elapsed}

# -----------------------------
# Expose /metrics endpoint
# -----------------------------
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest
    return PlainTextResponse(generate_latest())

"""