import streamlit as st
import time
from rag import add_document, generate_answer
from loaders import read_pdf
from timing import reset_timings

st.set_page_config(page_title="Local RAG MVP")
st.title("🧠 Local RAG (llama.cpp)")

uploaded_file = st.file_uploader(
    "Upload PDF or TXT", type=["pdf", "txt"]
)

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    if text.strip():
        reset_timings()
        add_document(text)
        st.success("Document indexed")

        if "timings" in st.session_state:
            st.markdown("### ⏱️ Indexing timings")
            st.json(st.session_state.timings)
    else:
        st.error("Could not extract text from file")

query = st.text_input("Ask a question")

if query:
    reset_timings()
    start = time.perf_counter()

    with st.spinner("Thinking..."):
        answer = generate_answer(query)

    total = (time.perf_counter() - start) * 1000

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### ⏱️ RAG timings")
    timings = dict(st.session_state.timings)
    timings["total"] = total
    st.json(timings)
