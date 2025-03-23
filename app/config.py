import os
from dotenv import load_dotenv
load_dotenv()

DENSE_ENCODER_PATH = os.getenv('DENSE_ENCODER_PATH')

DATASET = os.getenv('DATASET_PATH')

# Configurarea pentru Elasticsearch
class ESConfig:
    HOST = os.getenv('ES_HOST')
    BM25_INDEX = os.getenv('ES_INDEX')

class RedisConfig:
    HOST = os.getenv('REDIS_HOST')
    PORT = int(os.getenv('REDIS_PORT'))
    DB = 0
    EXPIRE_MS = 300

class ChromaConfig:
    PERSIST_DIRECTORY = "chroma_db"
    COLLECTION_NAME = "rag_dense"

class ModelConfig:
    MODEL_PATH = os.getenv('MODEL_PATH')
    N_CTX = 512
    N_GPU_LAYERS = -1
    N_THREADS = 8
    N_BATCH = 512
    USE_MMAP = True
    VERBOSE = False

class LLMConfig:
    INSTANCES = 2
    MAX_TOKENS = 112
    TEMPERATURE = 0.2
    TOP_P = 0.9
    STOP = ["<|eot_id|>", "\n\n"]

class UvicornConfig:
    APP = "app.main:app"
    HOST = os.getenv('UVICORN_HOST')
    PORT = int(os.getenv('UVICORN_PORT'))
    WORKERS = 1

# Obiectul ES cu configurarea
ES = ESConfig()
RC = RedisConfig()
CC = ChromaConfig()
MODEL_CONFIG = ModelConfig()
LLM_CONFIG = LLMConfig()
UV_CONFIG = UvicornConfig()