#LangChain RAG with Ollama
#pip install langchain langchain-community sentence-transformers ollama
#run ```ollama run llama3.2```


#create a file called rag_langchain_ollama.py

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Documents
docs = [
    "AWS provides cloud services.",
    "LangChain makes building LLM pipelines easier.",
    "Docker is used for containerized applications."
]

# 3. Store docs in FAISS
vectorstore = FAISS.from_texts(docs, embeddings)

# 4. Local LLM (via Ollama)
llm = Ollama(model="llama3.2")

# 5. RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# 6. Ask a question
query = "What helps simplify LLM pipelines?"
print("Query:", query)
print("Answer:", qa.run(query))



#and then run ```python rag_langchain_ollama.py```

#and should get output something like Query: What helps simplify LLM pipelines?
#Answer: LangChain makes building LLM pipelines easier.
#you’ve now built a local RAG pipeline with LangChain + Ollama.
#This is industry-level: you can swap FAISS for Pinecone, Ollama for OpenAI, with almost no changes