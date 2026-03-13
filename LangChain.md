# LangChain Modules & Imports Guide

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start-5-min)
- [Alternative Frameworks](#alternative-frameworks)
- [Module Structure](#module-structure-overview)
- [Import Breakdown](#import-breakdown)
- [LCEL Flow](#lcel-langchain-expression-language-flow)
- [Appendix I: Class Reference](#appendix-i--complete-class-reference)
- [Appendix II: Learning Resources](#appendix-ii-recommended-learning-resources)

<small><small><small>Last updated: March 2026</small></small></small>


---
**WARNING!** This is a copy for your convince for the maintained copy [**Click here**](https://gist.github.com/NicoJanE/2e4a9fcd8899c6ee2bc5363a0fb8c1d1.html)
<br>

## Introduction

This guide provides a comprehensive breakdown of **LangChain** — the Python framework that powers modern AI applications in combination with some general AI concepts . It explains every module, class, and concept used in building a Retrieval-Augmented Generation (RAG) agent that can answer questions using your own documents.

<details closed>  
  <summary class="clickable-summary">
    <span  class="summary-icon">  <b><u>For who?</u></b></span> 
 </summary>

### Who Should Read This?

- ✅ **AI/ML developers** building RAG or chatbot applications
- ✅ **Data engineers** integrating LLMs into pipelines
- ✅ **Students** learning modern AI frameworks
- ✅ **Anyone** confused by LangChain's module structure

</details>

<br>

### What is LangChain?

**LangChain** is an open-source Python framework that simplifies building applications with Large Language Models (LLMs). Instead of writing raw API calls, LangChain provides **pre-built components** that you pipe together to create intelligent workflows.

**Example workflow:**
```python
documents → chunks → embeddings → vector store → retriever → prompt → LLM → answer
```

LangChain handles each step; you just connect them with `|` (pipe).

**Key Features:**

- 🔗 **Chainable components** — Pipe workflows together with simple syntax
- 📄 **Document handling** — Load, split, and manage text from many sources
- 🧠 **Embeddings** — Convert text to vectors for semantic search
- 💾 **Vector stores** — Store and retrieve similar documents instantly
- 🤖 **LLM abstraction** — Use local (Ollama) or cloud (OpenAI) models interchangeably
- 📝 **Prompt templates** — Structure instructions for consistent LLM responses

<br>

### What is RAG? Why Use LangChain?

**The Problem**
Standard LLMs have **outdated knowledge** (training data is weeks/months old) and **no context** about your private documents.

**The Solution: RAG (Retrieval-Augmented Generation)**

1. **Index your documents** — Convert text into searchable vectors
2. **Retrieve relevant docs** — Find pieces matching the user's question
3. **Augment the prompt** — Add retrieved context before calling the LLM
4. **Generate answer** — LLM produces response using your documents + knowledge


<details closed>  
  <summary class="clickable-summary">
    <span  class="summary-icon">  <b><u>Use Cases</u></b></span> 
 </summary>

### Real-World Use Cases

Use cases for these techniques are:

1. **Internal Knowledge Base Bot**
  *"Answer questions about our company policies"*
  - Load internal docs → Index → Retrieve relevant sections → Answer

2. **Research Assistant**
  *"Summarize findings from PDFs"*
  - Load research papers → Split by topic → Find relevant → Generate summary

3. **Customer Support AI**
  *"Answer FAQ from knowledge base"*
  - Load FAQ docs → Embed → Retrieve matching Q&A → Respond

4. **Contract Analysis**
  *"Extract key terms from legal documents"*
  - Load contracts → Find clauses → Summarize terms

</details>

---

## Quick Start (5 min)

### Installation

For use in Python:

```bash
pip install langchain langchain-community langchain-text-splitters chromadb ollama tiktoken langchain-ollama
```

### Basic RAG Flow
```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load documents
documents = DirectoryLoader("docs").load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OllamaEmbeddings(model="mistral")

# 4. Store in vector database
db = Chroma.from_documents(chunks, embedding=embeddings)

# 5. Query
results = db.as_retriever().invoke("Your question here")
print(results)
```

### Next Steps
- See [Module Structure](#module-structure-overview) to understand each component
- Check [Import Breakdown](#import-breakdown) for detailed explanations
- Read [LCEL Flow](#lcel-langchain-expression-language-flow) to build complex chains

---

## Alternative Frameworks

LangChain is not the only RAG/LLM framework. Here's how it compares to competitors:

| Framework | Best For | Languages | Strengths | Weaknesses |
|-----------|----------|-----------|-----------|-----------|
| **LangChain** | General LLM apps + RAG | Python, JS, TypeScript | Largest ecosystem, many integrations, LCEL syntax | Steep learning curve, verbose |
| **LlamaIndex** | Document indexing + RAG | Python, TypeScript | Purpose-built for RAG, simple API, better doc handling | Smaller ecosystem, fewer LLM integrations |
| **Haystack** | Production RAG pipelines | Python | Designed for search/RAG, modular, good docs | Smaller community, fewer LLM models |
| **DSPy** | Prompt optimization | Python | Automatic prompt tuning, fewer tokens | Experimental, less mature |
| **Semantic Kernel** | Enterprise AI apps | Python, JS, TypeScript | Microsoft ecosystem, good for Azure | Newer, smaller community |
| **Embedchain** | Quick RAG prototypes | Python | Simple setup (5 lines of code), beginner-friendly | Very limited features, not scalable |
| **Verba** | Semantic search + RAG | Python, JS | Built on Weaviate, good UI, vector-native | Limited customization |

**Language availability note:** None of these frameworks have primary C/C++ or Rust support. **LangChain** and **Semantic Kernel** offer the best cross-language support with Python + TypeScript/JavaScript.

**For this project (RAG with Ollama):** LangChain offers the best balance of flexibility and ecosystem maturity. LlamaIndex would be simpler but offers fewer customization options.

---

## Module Structure Overview

```
langchain_core/
├── documents          → Document model for storing text + metadata
├── prompts           → Templates for structuring LLM inputs
└── runnables         → Chainable components (piping with |)

langchain_text_splitters/
└── RecursiveCharacterTextSplitter → Splits text into chunks

langchain_ollama/
└── OllamaEmbeddings  → Converts text → vectors (local)

langchain_community/
├── vectorstores      → Chroma (stores + retrieves embeddings)
└── llms              → Ollama (local LLM inference)
```

---

## Import Breakdown

### 1. **`from langchain_core.documents import Document`**

**What it does:** Container for text content plus metadata.

**Concept:** Data model that wraps document content with source information, page numbers, and custom metadata.

**Usage:**
```python
Document(
    page_content="The actual text content...",
    metadata={"source": "documents/file.md", "page": 1}
)
```

**Related Classes:**
- `BaseDocumentTransformer` — Base class for document transformations
- `Document` — The main document class

**Documentation:**
- https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html

---

### 2. **`from langchain_text_splitters import RecursiveCharacterTextSplitter`**

**What it does:** Splits long documents into smaller, manageable chunks.

**Concept:** Chunking — breaks large texts into pieces (default ~1000 chars) while maintaining semantic coherence. The "Recursive" part means it tries to split on meaningful boundaries (paragraphs, sentences) rather than arbitrary character positions.

**Usage:**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Size of each chunk
    chunk_overlap=0       # How much chunks overlap (prevents missing context)
)
chunks = text_splitter.split_documents(documents)
```

**Related Classes:**
- `CharacterTextSplitter` — Simple character-based splitting
- `TokenTextSplitter` — Splits by token count
- `SentenceTransformersTokenTextSplitter` — Semantic-aware splitting

**Documentation:**
- https://api.python.langchain.com/en/latest/text_splitters/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
- https://python.langchain.com/docs/modules/data_connection/document_transformers/

---

### 3. **`from langchain_ollama import OllamaEmbeddings`**

**What it does:** Converts text into numerical vectors (embeddings).

**Concept:** Embeddings — transforms words/sentences into dense vectors (usually 384 or 1024 dimensions). Words with similar meanings have similar vectors. This enables semantic search.

**Usage:**
```python
embeddings = OllamaEmbeddings(model="mistral")
vector = embeddings.embed_query("What is AI?")  # Returns [0.123, -0.456, ...]
```

**Related Classes:**
- `HuggingFaceEmbeddings` — Free embeddings from HuggingFace
- `OpenAIEmbeddings` — OpenAI's commercial embeddings
- `FakeEmbeddings` — For testing

**Documentation:**
- https://api.python.langchain.com/en/latest/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html
- https://python.langchain.com/docs/modules/data_connection/text_embedding/

---

### 4. **`from langchain_community.vectorstores import Chroma`**

**What it does:** Vector database that stores embeddings and enables similarity search.

**Concept:** Vector Store — persists embedded documents to disk, allowing fast retrieval of similar documents. Chroma is local and lightweight; alternatives include Pinecone (cloud) or Faiss (offline).

**Usage:**
```python
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
retriever = db.as_retriever(search_kwargs={"k": 4})  # Top 4 matches
```

**Related Classes:**
- `FAISS` — Facebook AI Similarity Search (offline, fast)
- `Pinecone` — Cloud-based vector database
- `Weaviate` — Open-source vector database
- `Milvus` — Self-hosted vector database

**Documentation:**
- https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html
- https://python.langchain.com/docs/modules/data_connection/vectorstores/

---

### 5. **`from langchain_community.llms import Ollama`**

**What it does:** Interface to Ollama (runs local LLMs without API keys).

**Concept:** LLM — Large Language Model interface. This wraps Ollama's HTTP API to generate text. Alternative: `langchain_ollama.OllamaLLM` (newer).

**Usage:**
```python
llm = Ollama(model="mistral")
response = llm.invoke("Explain quantum computing")
```

**Related Classes:**
- `OpenAI` — Uses OpenAI API
- `HuggingFaceHub` — Free HuggingFace models
- `Anthropic` — Claude models (requires API key)
- `LocalLLM` — Generic local model runner

**Documentation:**
- https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html
- https://python.langchain.com/docs/modules/model_io/llms/

---

### 6. **`from langchain_core.prompts import ChatPromptTemplate`**

**What it does:** Formats structured chat prompts (system + user messages).

**Concept:** Prompt Template — defines the structure of instructions sent to the LLM. Supports variables like `{context}` that get filled in dynamically.

**Usage:**
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Context: {context}"),
    ("human", "{input}")
])
```

**Related Classes:**
- `PromptTemplate` — Simple text template (non-chat)
- `FewShotChatMessagePromptTemplate` — Includes examples
- `MessagesPlaceholder` — Dynamic message insertion

**Documentation:**
- https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
- https://python.langchain.com/docs/modules/model_io/prompts/

---

### 7. **`from langchain_core.runnables import RunnablePassthrough`**

**What it does:** Passes input through unchanged (used in LCEL chains).

**Concept:** Runnables — building blocks that can be piped together with `|`. `RunnablePassthrough()` is a no-op that allows you to structure dict inputs.

**Usage:**
```python
qa_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
)
```

**Related Classes:**
- `RunnableParallel` — Run multiple chains in parallel
- `RunnableSequence` — Chain operations sequentially
- `RunnableLambda` — Wrap Python functions
- `RunnableMap` — Transform outputs

**Documentation:**
- https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html
- https://python.langchain.com/docs/expression_language/

---

## LCEL (LangChain Expression Language) Flow

The chain construction uses **LCEL** — a declarative syntax for composing components:

```python
qa_chain = (
    {
        # Step 1: Create a dict with two keys
        "context": retriever | format_docs,  # Retrieve docs and format
        "input": RunnablePassthrough()       # Pass query through unchanged
    }
    | prompt                                 # Step 2: Fill prompt template
    | llm                                    # Step 3: Run LLM
)
```

**Execution flow:**
1. Input: `"What is AI?"`
2. retriever finds similar docs → format_docs joins them
3. Creates dict: `{"context": "...", "input": "What is AI?"}`
4. prompt fills: `"You are helpful. Context: ...\n\nWhat is AI?"`
5. llm generates: `"AI is..."`

Think of it as **Unix pipes** 🔗:
```
input | retriever | format | fill_prompt | llm | output
```

---

## RAG Pipeline Visual Architecture

Here's how all the components work together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                │
└─────────────────────────────────────────────────────────────────────┘

INDEXING PHASE (One-time setup)
────────────────────────────────
    Documents (*.md)
           ↓
    ┌──────────────────────────┐
    │  DirectoryLoader         │  ← Load from files
    │  (or other loaders)      │
    └──────────────────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  RecursiveCharacterTextSplitter      │  ← Split into chunks
    │  (chunk_size=1000, overlap=0)        │
    └──────────────────────────────────────┘
           ↓
       Document Chunks
           ↓
    ┌──────────────────────────┐
    │  OllamaEmbeddings        │  ← Convert to vectors
    │  (model="mistral")       │    (384-1024 dimensions)
    └──────────────────────────┘
           ↓
       Embedding Vectors
           ↓
    ┌──────────────────────────────────────┐
    │  Chroma.from_documents()             │  ← Store vectors
    │  (persist_directory="chroma_db")     │
    └──────────────────────────────────────┘
           ↓
       Vector Database (ChromaDB)


QUERYING PHASE (Runtime)
────────────────────────
    User Query: "What is...?"
           ↓
    ┌──────────────────────────┐
    │  db.as_retriever()       │  ← Search vector DB
    │  (search_kwargs={"k": 4})│    Find top 4 matches
    └──────────────────────────┘
           ↓
       Retrieved Documents
           ↓
    ┌──────────────────────────┐
    │  format_docs()           │  ← Join chunks into context
    │  (join with "\n\n")      │
    └──────────────────────────┘
           ↓
       Formatted Context
           ↓
    ┌──────────────────────────────────────┐
    │  ChatPromptTemplate.from_messages()  │  ← Fill template
    │  ("Context: {context}\n{input}")     │
    └──────────────────────────────────────┘
           ↓
       Complete Prompt
           ↓
    ┌──────────────────────────┐
    │  Ollama(model="mistral") │  ← Generate response
    │  (local LLM inference)   │
    └──────────────────────────┘
           ↓
       LLM Response (Answer)
```

**Key Points:**
- **Indexing** happens once when you load documents
- **Retrieval** uses vector similarity (fast, semantic)
- **Augmentation** adds context to the prompt
- **Generation** LLM creates answer with context

**Documentation:**
- https://python.langchain.com/docs/expression_language/

---

<details closed>  
  <summary class="clickable-summary">
    <span  class="summary-icon">  <b><u>Appendix I</u></b>: Complete Class Reference</span> 
 </summary>

### langchain_core.documents
| Class | Purpose |
|-------|---------|
| `Document` | Text + metadata container |
| `BaseDocumentTransformer` | Base for document transformations |

### langchain_text_splitters
| Class | Purpose |
|-------|---------|
| `RecursiveCharacterTextSplitter` | Smart recursive splitting |
| `CharacterTextSplitter` | Simple splitting by character |
| `TokenTextSplitter` | Splitting by token count |
| `SentenceTransformersTokenTextSplitter` | Token-aware semantic splitting |

### langchain_ollama
| Class | Purpose |
|-------|---------|
| `OllamaEmbeddings` | Embed text with Ollama |
| `OllamaLLM` | Generate text with Ollama |

### langchain_community.vectorstores
| Class | Purpose |
|-------|---------|
| `Chroma` | Local vector database |
| `FAISS` | Offline similarity search |
| `Pinecone` | Cloud vector database |
| `Weaviate` | Self-hosted vector DB |
| `Milvus` | Enterprise vector DB |

### langchain_community.llms
| Class | Purpose |
|-------|---------|
| `Ollama` | Local LLM (via Ollama) |
| `OpenAI` | OpenAI API models |
| `HuggingFaceHub` | HuggingFace models |
| `Anthropic` | Claude models |

### langchain_core.prompts
| Class | Purpose |
|-------|---------|
| `ChatPromptTemplate` | Chat message template |
| `PromptTemplate` | Text template |
| `FewShotChatMessagePromptTemplate` | Template with examples |

### langchain_core.runnables
| Class | Purpose |
|-------|---------|
| `RunnablePassthrough` | Pass through input |
| `RunnableParallel` | Run in parallel |
| `RunnableSequence` | Chain sequentially |
| `RunnableLambda` | Wrap Python functions |

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation — answer questions using documents |
| **Embedding** | Numerical vector representation of text (captures meaning) |
| **Chunking** | Breaking documents into smaller overlapping pieces |
| **Vector Store** | Database of embeddings enabling similarity search |
| **LLM** | Large Language Model (e.g., Mistral, GPT-4) |
| **Prompt Template** | Structured instruction format for LLMs |
| **LCEL** | LangChain Expression Language (piping syntax with `\|`) |
| **Retriever** | Component that finds relevant documents for a query |

---

</details>

### Official Documentation
- **LangChain Docs (main):** https://python.langchain.com/docs/
- **LangChain API Reference:** https://api.python.langchain.com/
- **LangChain GitHub:** https://github.com/langchain-ai/langchain

### Specific Topics
- **RAG Overview:** https://python.langchain.com/docs/use_cases/question_answering/
- **Document Loaders:** https://python.langchain.com/docs/modules/data_connection/document_loaders/
- **Text Splitters:** https://python.langchain.com/docs/modules/data_connection/document_transformers/
- **Embeddings:** https://python.langchain.com/docs/modules/data_connection/text_embedding/
- **Vector Stores:** https://python.langchain.com/docs/modules/data_connection/vectorstores/
- **LLMs:** https://python.langchain.com/docs/modules/model_io/llms/
- **LCEL:** https://python.langchain.com/docs/expression_language/

### Related Tools
- **Ollama:** https://ollama.ai/
- **Chroma:** https://www.trychroma.com/
- **ChromaDB Docs:** https://docs.trychroma.com/

### Learning Paths
- **LangChain Quickstart:** https://python.langchain.com/docs/get_started/quickstart
- **LangChain Tutorials:** https://python.langchain.com/docs/tutorials/
- **YouTube — LangChain Crash Course:** Search "LangChain crash course"

</details>

<details closed>  
  <summary class="clickable-summary">
    <span  class="summary-icon">  <b><u>Appendix III</u></b>: Quick Setup Reminder</span> 
 </summary>

## Quick Setup Reminder

```bash
pip install langchain langchain-community langchain-text-splitters chromadb ollama tiktoken langchain-ollama

# Then ensure Ollama is running
ollama serve

# In another terminal, pull a model
ollama pull mistral

# Run the agent
python Ollama-RAG-Agent.py
```

</details>

---

**Last Updated:** March 2026  
**Version:** LangChain 0.2+
