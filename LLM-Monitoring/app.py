from fastapi import FastAPI, Query
from langfuse import Langfuse
import time
import random

app = FastAPI(title="LLM Monitoring Demo")

# -----------------------------
# Initialize Langfuse client
# -----------------------------
langfuse = Langfuse(
    secret_key="dev",  # default for local dev
    public_key="dev"
)

@app.get("/ask")
def ask(q: str = Query(...)):
    start = time.time()

    # fake answer (in real app, call LLM)
    answer = f"Answer to '{q}' is {random.choice(['42', 'cloud computing', 'use Docker'])}"

    latency = round(time.time() - start, 2)

    # -----------------------------
    # Send event to Langfuse
    # -----------------------------
    trace = langfuse.trace(name="rag-query", input=q, output=answer, latency=latency)

    return {"question": q, "answer": answer, "latency_sec": latency}
