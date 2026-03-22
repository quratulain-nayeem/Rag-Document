import os
import fitz  # this is PyMuPDF - fitz is just its internal name
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from config import *

# ── 1. PDF LOADING ──────────────────────────────────────────
def load_pdf(file_path):
    """
    Opens a PDF and extracts all text from every page.
    Returns a list of Document objects (one per page).
    """
    doc = fitz.open(file_path)
    documents = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # skip empty pages
            documents.append(Document(
                page_content=text,
                metadata={"page": page_num + 1, "source": file_path}
            ))

    return documents

# ── 2. CHUNKING ──────────────────────────────────────────────
def chunk_documents(documents):
    """
    Splits pages into smaller chunks.
    Why: A full page is too long to embed meaningfully.
    Smaller chunks = more precise retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# ── 3. EMBEDDINGS + VECTOR STORE ─────────────────────────────
def build_vectorstore(chunks):
    """
    Converts chunks into numbers (embeddings) and stores them in ChromaDB.
    Why embeddings: Computers can't search text by meaning, only by numbers.
    Embeddings turn meaning into numbers so we can do semantic search.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    return vectorstore

# ── 4. RETRIEVAL ──────────────────────────────────────────────
def get_retriever(vectorstore):
    """
    Creates a retriever — something that takes a question
    and returns the TOP_K most relevant chunks.
    """
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ── 5. LLM CONNECTION ─────────────────────────────────────────
def get_llm():
    print(f"API KEY LOADED: {GROQ_API_KEY}")  # temporary debug line
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )
# ── 6. THE FULL RAG PIPELINE ──────────────────────────────────
def answer_question(question, retriever, llm):
    # Step 1: retrieve relevant chunks
    relevant_chunks = retriever.invoke(question)

    # Step 2: build context string from chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # Step 3: sources so user knows where answer came from
    sources = [f"Page {chunk.metadata['page']}" for chunk in relevant_chunks]

    # Step 4: craft the prompt
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I cannot find this in the provided document."

Context:
{context}

Question: {question}

Answer:"""

    # Step 5: send to LLM and get response
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content, list(set(sources))
    # Step 1: retrieve relevant chunks
    relevant_chunks = retriever.invoke(question)

    # Step 2: build context