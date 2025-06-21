# app/models.py
import os
import time
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
from app.config import MODEL_CONFIG, DENSE_ENCODER_PATH, LLM_CONFIG
import torch

# Detectează dispozitivul o singură dată la început
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")

try:
    from sentence_transformers import SentenceTransformer
    dense_encoder = SentenceTransformer(DENSE_ENCODER_PATH)
    dense_encoder = dense_encoder.to(device)
    print(f"Dense encoder loaded on device: {device}")
except Exception as e:
    print(f"Dense encoder initialization error: {str(e)}")
    dense_encoder = None

# Configurare LLM – pool de instanțe Llama
llama_pool = []
for _ in range(LLM_CONFIG.INSTANCES):
    if os.path.exists(MODEL_CONFIG.MODEL_PATH):
        llm_instance = Llama(
            model_path=MODEL_CONFIG.MODEL_PATH,
            n_ctx=MODEL_CONFIG.N_CTX,
            n_gpu_layers=MODEL_CONFIG.N_GPU_LAYERS,
            n_threads=MODEL_CONFIG.N_THREADS,
            n_batch=MODEL_CONFIG.N_BATCH,
            use_mmap=MODEL_CONFIG.USE_MMAP,
            verbose=MODEL_CONFIG.VERBOSE,
            device_map=device
        )
        llama_pool.append(llm_instance)
    else:
        raise FileNotFoundError("Llama model file missing!")

executor = ThreadPoolExecutor(max_workers=LLM_CONFIG.INSTANCES)
pool_index = 0

def generate_answer(prompt: str):
    global pool_index
    start = time.time()
    llm_instance = llama_pool[pool_index]
    pool_index = (pool_index + 1) % len(llama_pool)
    future = executor.submit(llm_instance, prompt, max_tokens=LLM_CONFIG.MAX_TOKENS, temperature=LLM_CONFIG.TEMPERATURE, top_p=LLM_CONFIG.TOP_P, stop=LLM_CONFIG.STOP)
    result = future.result()
    generation_time = time.time() - start
    answer = result['choices'][0]['text'].strip() if result and 'choices' in result else ""
    return answer, generation_time
