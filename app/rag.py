import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama
from timing import UITimer

@st.cache_resource
def load_llm():
    return Llama(
        model_path="/models/mistral.gguf",
        n_ctx=8192,
    )

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    return faiss.IndexFlatL2(384)

llm = load_llm()
embedding_model = load_embedding_model()
index = load_index()

if "documents" not in st.session_state:
    st.session_state.documents = []

def add_document(text: str):
    with UITimer("embed(document)"):
        emb = embedding_model.encode([text])

    with UITimer("faiss.add"):
        index.add(emb)

    st.session_state.documents.append(text)

def retrieve(query: str, k: int = 3):
    with UITimer("embed(query)"):
        q_emb = embedding_model.encode([query])

    with UITimer("faiss.search"):
        _, ids = index.search(q_emb, k)

    docs = st.session_state.documents
    retrieved = [docs[i] for i in ids[0] if i < len(docs)]

    if not retrieved:
        raise ValueError("No valid documents found")

    return retrieved

def generate_answer(query: str) -> str:
    with UITimer("retrieve"):
        context_docs = retrieve(query)

    with UITimer("prompt.build"):
        context = "\n\n".join(context_docs)
        prompt = f"""You are a helpful assistant.
Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer:"""

    with UITimer("llm.inference"):
        output = llm(
            prompt,
            max_tokens=300,
            temperature=0.2,
            stop=["</s>"],
        )

    return output["choices"][0]["text"].strip()
