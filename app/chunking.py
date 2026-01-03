from typing import List

def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 256,
    overlap: int = 40,
) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        step = max(1, max_tokens - overlap)
        start += step
        
        if end == len(tokens):
            break

    return chunks