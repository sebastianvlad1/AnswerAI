# app/storage.py
import json
import time
import numpy as np
import redis
from elasticsearch import Elasticsearch
import chromadb
from chromadb.config import Settings
from app.utils import clean_text, load_custom_data
from app.config import ES, RC, CC

# Redis pentru caching
redis_client = redis.Redis(host=RC.HOST, port=RC.PORT, db=RC.DB)

# Elasticsearch pentru BM25
es = Elasticsearch(hosts=[ES.HOST])

chroma_client = chromadb.Client(settings=Settings(
    persist_directory=CC.PERSIST_DIRECTORY
))
chroma_collection = chroma_client.get_or_create_collection(CC.COLLECTION_NAME)

# Load data
CUSTOM_DATA = load_custom_data()  # Awaiting a list of documents

# Construiește indexul Elasticsearch dacă nu există
def build_es_index():
    if not es.indices.exists(index=ES.BM25_INDEX):
        mapping = {
            "mapping": {
              "type": "text"
            }
        }
        es.indices.create(index=ES.BM25_INDEX, body=mapping)
        for idx, doc in enumerate(CUSTOM_DATA):
            es.index(index=ES.BM25_INDEX, id=idx, body={"text": doc})
build_es_index()

def index_chroma():
    docs = []
    embeddings = []
    metadatas = []
    ids = []
    from app.models import dense_encoder  # Import here to avoid circular dependencies
    for item in CUSTOM_DATA:
        docs.append(item["text"])
        ids.append(item["id"])
        if dense_encoder:
            emb = dense_encoder.encode(clean_text(item["text"])).tolist()
            print(emb)
            # Ensure the embedding is not empty
            if not emb:
                emb = [0.0]  # Fallback: a 1-dimensional zero vector; adjust dimension as needed
        else:
            # Fallback embedding (e.g., a vector of zeros with a predefined dimension, e.g. 768)
            emb = [0.0] * 768
        embeddings.append(emb)
        metadatas.append({"id": item["id"]})
    chroma_collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)

index_chroma()

def hybrid_retrieve(query: str, top_k=3, bm25_top=7):
    start = time.time()
    cleaned_query = clean_text(query)
    cache_key = f"hybrid:{cleaned_query}"
    cached = redis_client.get(cache_key)
    if cached:
        docs = json.loads(cached)
        return docs, time.time() - start

    # BM25 retrieval cu Elasticsearch
    es_result = es.search(
        index=ES.BM25_INDEX,
        body={
            "query": {"match": {"text": cleaned_query}},
            "size": bm25_top
        }
    )
    bm25_ids = [int(hit["_id"]) for hit in es_result["hits"]["hits"]]

    # Dense retrieval cu ChromaDB
    from app.models import dense_encoder
    if dense_encoder:
        query_embedding = dense_encoder.encode(cleaned_query).tolist()
        dense_result = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=bm25_top,
            include=["metadatas", "documents", "distances"]
        )
        dense_ids = [int(item["id"]) for item in dense_result["metadatas"][0]]
    else:
        dense_ids = []

    # Combină rezultate: folosim unirea simplă a id-urilor
    combined_ids = list(set(bm25_ids + dense_ids))
    # Selectează primele top_k documente
    selected_docs = [CUSTOM_DATA[i]["text"] for i in combined_ids][:top_k]

    # Salvează în cache (5 minute)
    redis_client.set(cache_key, json.dumps(selected_docs), ex=RC.EXPIRE_MS)
    return selected_docs, time.time() - start

def build_prompt(query: str, context_docs: list) -> str:
    # Pentru simplitate, concatenăm documentele (în producție, folosește un text splitter mai avansat)
    print(context_docs)
    context = " ".join(context_docs)
    return f"Context: {context}\nQuestion: {query}\nAnswer:"
