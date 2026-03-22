# 🧠 PaperBrain

> Upload a document. Ask anything. Answers grounded 
> in your file — not AI memory.

PaperBrain is a production-grade RAG (Retrieval Augmented 
Generation) system that lets you upload any PDF and ask 
questions about it. Every answer is grounded in your 
document with source citations — no hallucination, 
no guessing.

## How it works

1. Upload a PDF → PyMuPDF extracts text page by page
2. Text is split into 500-token chunks (100-token overlap)
3. Chunks are converted to embeddings using all-MiniLM-L6-v2
4. Embeddings stored in ChromaDB vector database
5. Your question is embedded and matched to top 5 chunks
6. Groq's Llama 3.3 70b answers using only those chunks

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Gradio |
| PDF Parsing | PyMuPDF |
| Chunking | LangChain Text Splitters |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| LLM | Groq API (Llama 3.3 70b) |
| Deployment | Render |

## Setup

1. Clone the repo
2. Install dependencies
3. Add your Groq API key to .env
4. Run the app

## Environment Variables

GROQ_API_KEY=your_key_here

## Local Development

py app.py

## Deployment

Deployed on Render — always live, free tier.

## What is RAG?

RAG stands for Retrieval Augmented Generation. 
Instead of relying on an AI's training data 
(which can be outdated or hallucinated), RAG 
retrieves relevant pieces of your actual document 
and uses those as context for the answer. 
The AI is grounded in your file, nothing else.

## Built by

Quratulain Nayeem
