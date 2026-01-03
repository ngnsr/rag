import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama
from timing import UITimer
from chunking import chunk_text
import os

MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf" 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_llm():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Please download Llama 3.2 3B to {MODEL_PATH}")
    
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        n_batch=1024,
        n_threads=8,
        use_mmap=True,
        verbose=True,
    )

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_index():
    return faiss.IndexFlatL2(384)

llm = load_llm()
embedding_model = load_embedding_model()
index = load_index()

def add_document(text: str):
    tokenizer = embedding_model.tokenizer
    with UITimer("chunking"):
        chunks = chunk_text(text, tokenizer, max_tokens=250, overlap=30)
    with UITimer("embed(chunks)"):
        embeddings = embedding_model.encode(chunks)
    with UITimer("faiss.add"):
        index.add(embeddings)
    st.session_state.chunks.extend(chunks)

def retrieve(query: str, k: int = 5, max_context_tokens: int = 2000):
    with UITimer("embed(query)"):
        q_emb = embedding_model.encode([query])
    with UITimer("faiss.search"):
        _, ids = index.search(q_emb, k=15) 

    chunks_data = st.session_state.chunks
    selected_chunks = []
    current_token_count = 0
    tokenizer = embedding_model.tokenizer

    for i in ids[0]:
        if i < 0 or i >= len(chunks_data): continue
        chunk = chunks_data[i]
        chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)
        if current_token_count + len(chunk_tokens) > max_context_tokens: continue 
        selected_chunks.append(chunk)
        current_token_count += len(chunk_tokens)
        
    return selected_chunks

def generate_answer(query: str) -> str:
    MAX_CONTEXT_TOKENS = 3000

    with UITimer("retrieve"):
        context_chunks = retrieve(query, k=6, max_context_tokens=MAX_CONTEXT_TOKENS)

    if not context_chunks:
        return "I couldn't find relevant information."

    with UITimer("prompt.build"):
        context_text = "\n\n".join(context_chunks)
        
        # --- LLAMA 3 PROMPT ---
        # Fixed: Removed manual <|begin_of_text|> to avoid duplicates
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context_text}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        # --- DEBUG PRINT ---
        print("\n" + "="*30 + " PROMPT START " + "="*30)
        print(prompt)
        print("="*31 + " PROMPT END " + "="*31 + "\n")

    with UITimer("llm.inference"):
        output = llm(
            prompt,
            max_tokens=512,
            temperature=0.1,
            stop=["<|eot_id|>"], 
            echo=False
        )

    return output["choices"][0]["text"].strip()