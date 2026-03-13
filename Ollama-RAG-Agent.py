"""
ATS Ollama RAG Agent - RAG Implementation using Ollama and ChromaDB

REQUIREMENTS
1.  Install required Python packages using pip:
        pip install langchain langchain-community langchain-text-splitters chromadb ollama, tiktoken 
        pip install langchain-ollama
        
2.  Ensure Ollama is running locally (default: http://localhost:11434)

IMPORTS
- langchain_community: Community-maintained integrations for document loaders, embeddings, vectorstores, and LLMs
- langchain: Core chain and splitter functionality
- chromadb: Vector database for storing and retrieving embeddings
- ollama: Ollama client for local LLM inference


Use:
- ollama list                       # to display the Models installed
- ollama pull mistral          # to get the model 
- python Ollama-RAG-Agent.py     # Run the program
 
"""

import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configuration
OLLAMA_MODEL = "mistral"                    # Or your preferred Ollama model
EMBEDDING_MODEL = "mistral"              # Must match OLLAMA_MODEL for simplicity now.
DOCUMENT_DIR = "RAG_documents"     # Directory containing your Markdown files
PERSIST_DIRECTORY = "chroma_db"        # Where ChromaDB will store the index

# 1. Load Documents
def load_markdown_files(directory):
    """Load all markdown files from directory"""
    documents = []
    doc_dir = Path(directory)
    if not doc_dir.exists():
        print(f"Warning: Directory '{directory}' does not exist. Creating empty document list.")
        return documents
    
    for md_file in doc_dir.rglob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(Document(page_content=content, metadata={"source": str(md_file)}))
    
    print(f"Loaded {len(documents)} markdown files")
    return documents

documents = load_markdown_files(DOCUMENT_DIR)

# 2. Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

# 3. Create Embeddings and Store in ChromaDB
#    - Configure OllamaEmbeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

#    - Create ChromaDB (or load existing)
db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)

# 4. Initialize the LLM and Retrieval Chain
llm = OllamaLLM(model=OLLAMA_MODEL)
retriever = db.as_retriever(search_kwargs={"k": 4})

# Create a prompt template for the chain
system_prompt = """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Format documents for context
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Create the chain using LCEL (LangChain Expression Language)
qa_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
)

# 5. Query Time!
query = "What is the main topic of doc1.md?"
result = qa_chain.invoke(query)
print(result)
