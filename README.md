# 🤖 Smart Document Chat - RAG

An AI-powered document question-answering application built using
Python, Streamlit, LangChain, FAISS, Hugging Face, and RAG.

The application allows users to upload a PDF document and ask
questions about its content. Relevant document sections are retrieved
using vector similarity search and provided to the AI model to generate
an answer.

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Semantic document search
- 🤖 AI-powered question answering
- 🧠 Retrieval-Augmented Generation (RAG)
- ⚡ FAISS vector database for fast similarity search
- 🔗 LangChain document processing
- ☁️ Hugging Face AI model integration
- 📊 Document word-cloud visualization
- 💬 Interactive Streamlit chat interface
- 🔐 API key stored securely using Streamlit secrets

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face
- Sentence Transformers
- PyPDF
- Matplotlib
- WordCloud

## 🏗️ Project Architecture

```text
PDF Document
     ↓
PyPDFLoader
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Hugging Face Embeddings
     ↓
FAISS Vector Database
     ↓
User Question
     ↓
Similarity Search
     ↓
Relevant Document Chunks
     ↓
Hugging Face LLM
     ↓
Generated Answer