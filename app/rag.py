from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama

# Embeddings
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

DIM = 384
index = faiss.IndexFlatL2(DIM)
documents: list[str] = []

# LLM (llama.cpp)
llm = Llama(model_path="/models/mistral.gguf", n_ctx=8192)

def add_document(text: str):
    emb = embedding_model.encode([text])
    index.add(emb)
    documents.append(text)

def retrieve(query: str, k: int = 3):
    q_emb = embedding_model.encode([query])
    _, ids = index.search(q_emb, k)
    retrieved = [documents[i] for i in ids[0] if i < len(documents)]
    if not retrieved:
        raise ValueError(f"No valid documents found for query: {query}")
    return retrieved

def generate_answer(query: str) -> str:
    context = "\n\n".join(retrieve(query))

    prompt = f"""You are a helpful assistant.
Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer:"""

    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.2,
        stop=["</s>"]
    )

    return output["choices"][0]["text"].strip()
