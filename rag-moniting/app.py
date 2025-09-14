import time
import logging
from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

from langfuse import Langfuse
import mlflow

# For demo RAG
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="RAG with Langfuse + MLflow")

# -----------------------------
# Monitoring (Langfuse + MLflow)
# -----------------------------
langfuse = Langfuse(secret_key="dev", public_key="dev")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rag-demo")

REQUEST_COUNT = Counter("rag_requests_total", "Total RAG requests")
REQUEST_LATENCY = Histogram("rag_request_latency_seconds", "Latency of RAG requests")

# -----------------------------
# Configure LLM + Embeddings
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding("all-MiniLM-L6-v2")
Settings.llm = Ollama(model="llama3.2")

# -----------------------------
# Load docs + build FAISS index
# -----------------------------
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, storage_context=FaissVectorStore.from_documents(documents))
query_engine = index.as_query_engine()

# -----------------------------
# API endpoint
# -----------------------------
@app.get("/ask")
@REQUEST_LATENCY.time()
def ask_question(q: str = Query(...), request: Request = None):
    REQUEST_COUNT.inc()
    start = time.time()

    response = query_engine.query(q)
    elapsed = round(time.time() - start, 2)

    # -----------------------------
    # 1. Log to Langfuse
    # -----------------------------
    langfuse.trace(
        name="rag-query",
        input=q,
        output=str(response),
        latency=elapsed
    )

    # -----------------------------
    # 2. Log to MLflow
    # -----------------------------
    with mlflow.start_run():
        mlflow.log_param("query", q)
        mlflow.log_metric("latency", elapsed)
        mlflow.log_metric("answer_length", len(str(response)))
        mlflow.log_text(str(response), "answer.txt")

    # -----------------------------
    # 3. Log locally
    # -----------------------------
    client_ip = request.client.host if request else "unknown"
    logger.info(f"Client {client_ip} | Q: {q} | A: {response} | Time: {elapsed}s")

    return {"question": q, "answer": str(response), "latency_sec": elapsed}

# -----------------------------
# Prometheus endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())
