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

def _get_doc_id(item, fallback_index: int) -> str:
    if isinstance(item, dict) and item.get("id") is not None:
        return str(item["id"])
    return str(fallback_index)

def _get_doc_text(item) -> str:
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(item)

def _iter_documents():
    for idx, item in enumerate(CUSTOM_DATA):
        yield _get_doc_id(item, idx), _get_doc_text(item)

def _document_lookup() -> dict:
    return {doc_id: text for doc_id, text in _iter_documents()}

def _doc_id_from_es_hit(hit) -> str:
    source = hit.get("_source") or {}
    if source.get("doc_id") is not None:
        return str(source["doc_id"])

    text_source = source.get("text")
    if isinstance(text_source, dict) and text_source.get("id") is not None:
        return str(text_source["id"])

    return str(hit["_id"])

def _merge_ranked_ids(*ranked_id_lists) -> list:
    ordered_ids = []
    seen = set()
    for ranked_ids in ranked_id_lists:
        for doc_id in ranked_ids:
            doc_id = str(doc_id)
            if doc_id not in seen:
                ordered_ids.append(doc_id)
                seen.add(doc_id)
    return ordered_ids

# Construiește indexul Elasticsearch dacă nu există
def build_es_index():
    if not es.indices.exists(index=ES.BM25_INDEX):
        mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"}
                }
            }
        }
        es.indices.create(index=ES.BM25_INDEX, body=mapping)
        for doc_id, text in _iter_documents():
            es.index(index=ES.BM25_INDEX, id=doc_id, body={"doc_id": doc_id, "text": text})
build_es_index()

def index_chroma():
    docs = []
    embeddings = []
    metadatas = []
    ids = []
    from app.models import dense_encoder  # Import here to avoid circular dependencies
    for doc_id, text in _iter_documents():
        docs.append(text)
        ids.append(doc_id)
        if dense_encoder:
            emb = dense_encoder.encode(clean_text(text)).tolist()
            # Ensure the embedding is not empty
            if not emb:
                emb = [0.0]  # Fallback: a 1-dimensional zero vector; adjust dimension as needed
        else:
            # Fallback embedding (e.g., a vector of zeros with a predefined dimension, e.g. 768)
            emb = [0.0] * 768
        embeddings.append(emb)
        metadatas.append({"id": doc_id})
    if docs:
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
    bm25_ids = [_doc_id_from_es_hit(hit) for hit in es_result["hits"]["hits"]]

    # Dense retrieval cu ChromaDB
    from app.models import dense_encoder
    if dense_encoder:
        query_embedding = dense_encoder.encode(cleaned_query).tolist()
        dense_result = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=bm25_top,
            include=["metadatas", "documents", "distances"]
        )
        dense_ids = [str(item["id"]) for item in dense_result["metadatas"][0] if item and item.get("id") is not None]
    else:
        dense_ids = []

    # Combină rezultate păstrând ordinea de ranking: BM25 apoi dense, fără duplicate.
    combined_ids = _merge_ranked_ids(bm25_ids, dense_ids)
    # Selectează primele top_k documente
    doc_lookup = _document_lookup()
    selected_docs = [doc_lookup[doc_id] for doc_id in combined_ids if doc_id in doc_lookup][:top_k]

    # Salvează în cache (5 minute)
    redis_client.set(cache_key, json.dumps(selected_docs), ex=RC.EXPIRE_MS)
    return selected_docs, time.time() - start

def build_prompt(query: str, context_docs: list) -> str:
    # Pentru simplitate, concatenăm documentele (în producție, folosește un text splitter mai avansat)
    print(context_docs)
    context = " ".join(context_docs)
    return f"Context: {context}\nQuestion: {query}\nAnswer:"
