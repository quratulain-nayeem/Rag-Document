import os
import tempfile

import streamlit as st
from rag import load_pdf, chunk_documents, build_vectorstore, get_retriever, get_llm, answer_question

st.set_page_config(page_title="PaperBrain", page_icon="🧠")
st.title("🧠 PaperBrain")
st.caption("Upload a document. Ask anything. Answers grounded in your file — not AI memory.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Processing..."):
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                documents = load_pdf(tmp_path)
                chunks = chunk_documents(documents)
                vectorstore = build_vectorstore(chunks)
                st.session_state.retriever = get_retriever(vectorstore)
                st.session_state.llm = get_llm()
                st.success(f"Done! {len(chunks)} chunks created.")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

question = st.text_input("Ask a question")
if st.button("Ask") and question:
    if "retriever" not in st.session_state:
        st.warning("Please upload and process a PDF first.")
    else:
        if "llm" not in st.session_state:
            st.session_state.llm = get_llm()
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(
                    question,
                    st.session_state.retriever,
                    st.session_state.llm,
                )
            except Exception as e:
                st.error(str(e))
            else:
                st.write(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
