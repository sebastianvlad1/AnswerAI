# app/utils.py
import re
import json
import logging
from app.config import DATASET

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return re.sub(r'[^\w\s-]', '', text)

def load_custom_data() -> list:
    try:
        with open(DATASET, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error("Error loading custom data: " + str(e))
        data = []
    return data

def log_query(query: str, retrieval_time: float, generation_time: float):
    logging.info(f"Query: {query} | Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s")
