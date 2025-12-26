import streamlit as st
from rag import add_document, generate_answer
from loaders import read_pdf

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
        add_document(text)
        st.success("Document indexed")
    else:
        st.error("Could not extract text from file")

query = st.text_input("Ask a question")

if query:
    with st.spinner("Thinking..."):
        answer = generate_answer(query)
    st.markdown("### Answer")
    st.write(answer)
