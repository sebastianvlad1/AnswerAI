# app/rag.py
from app.storage import hybrid_retrieve, build_prompt
from app.models import generate_answer
from app.utils import log_query


def process_query(query: str, top_k=3):
    # 1. Retrieval hibrid (Elasticsearch + ChromaDB, cu caching Redis)
    context_docs, retrieval_time = hybrid_retrieve(query, top_k=top_k)

    # 2. Construiește prompt-ul pentru LLM
    prompt = build_prompt(query, context_docs)

    # 3. Generare răspuns folosind pool-ul de instanțe LLM
    answer, generation_time = generate_answer(prompt)

    log_query(query, retrieval_time, generation_time)
    return {
        "answer": answer,
        "context": context_docs,
        "timing": {
            "retrieval": retrieval_time,
            "generation": generation_time
        }
    }
