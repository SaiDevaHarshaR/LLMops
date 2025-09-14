#`ollama pull llama3.2` and then run the command `ollama run llama3.2`
#install all the python dependencies `pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface sentence-transformers`
 #the following is the code for running finetuning with RAG llamaIndex
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Choose an embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# 2. Connect to local Ollama model
Settings.llm = Ollama(model="llama3.2")

# 3. Create some small knowledge base (documents)
texts = [
    "AWS provides many cloud services like EC2, S3, and Lambda.",
    "LangChain helps developers build LLM-powered applications.",
    "Docker is used to package applications into containers."
]
docs = [Document(text=t) for t in texts]

# 4. Build a vector index from documents
index = VectorStoreIndex.from_documents(docs)

# 5. Create a query engine (to ask questions)
query_engine = index.as_query_engine()

# 6. Ask a question
question = "What helps simplify building LLM applications?"
response = query_engine.query(question)

# 7. Print answer
print("Question:", question)
print("Answer:", str(response))

#and run the code `python LlamaIndex RAG with Ollama.py` 
#Why? We’re importing the Lego blocks:

#VectorStoreIndex → builds a searchable vector database (your “memory”).

#Document → wraps plain text into a format LlamaIndex can use.

#Settings → set defaults for which embedding model & LLM we want.

#Ollama → lets LlamaIndex talk to Ollama (our local LLM).

#HuggingFaceEmbedding → turns text into embeddings (numbers).
