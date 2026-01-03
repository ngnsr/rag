import streamlit as st
import time
import os
from rag import add_document, generate_answer
from loaders import read_pdf
from timing import reset_timings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Force Initialization of Session State ---
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "timings" not in st.session_state:
    st.session_state.timings = {}

st.set_page_config(page_title="Gemma RAG MVP")
st.title("💎 Local RAG (Gemma 3B)")

uploaded_file = st.file_uploader(
    "Upload PDF or TXT", type=["pdf", "txt"]
)

if uploaded_file:
    with st.spinner("Processing document..."):
        if uploaded_file.type == "application/pdf":
            text = read_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        if text.strip():
            reset_timings()
            add_document(text)
            st.success(f"Document indexed ({len(text)} chars)")

            if "timings" in st.session_state:
                st.markdown("### ⏱️ Indexing timings")
                st.json(st.session_state.timings)
        else:
            st.error("Could not extract text from file")

query = st.text_input("Ask a question about the document")

if query:
    reset_timings()
    start = time.perf_counter()

    with st.spinner("Gemma is thinking..."):
        answer = generate_answer(query)

    total = (time.perf_counter() - start) * 1000

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### ⏱️ RAG timings")
    timings = dict(st.session_state.timings)
    timings["total_ms"] = total
    st.json(timings)