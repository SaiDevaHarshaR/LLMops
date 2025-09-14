# pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

# 1. Embedding model (HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Store docs in FAISS
docs = [
    "AWS provides cloud services.",
    "LangChain simplifies LLM pipelines.",
    "Docker is for containerized apps."
]
vectorstore = FAISS.from_texts(docs, embeddings)

# 3. LLM (local LLaMA via Ollama)
llm = ChatOllama(model="llama2", temperature=0)  # you can swap to "mistral", "codellama", etc.

# 4. RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# 5. Ask a question
query = "What helps simplify LLM pipelines?"
print(qa.run(query))
