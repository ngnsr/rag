from llama_cpp import Llama
import time

# Initialize
llm = Llama(
    model_path="models/llama-3.2-3b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1, # Metal
    n_ctx=2048,
    verbose=False
)

prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nCount to 50.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

print("WARMUP (Loading model into GPU)...")
llm(prompt, max_tokens=10)

print("STARTING SPEED TEST...")
start = time.perf_counter()
output = llm(prompt, max_tokens=200)
end = time.perf_counter()

tokens = output["usage"]["completion_tokens"]
duration = end - start
tps = tokens / duration

print(f"Generated {tokens} tokens in {duration:.2f} seconds")
print(f"⚡ SPEED: {tps:.2f} Tokens/Sec")