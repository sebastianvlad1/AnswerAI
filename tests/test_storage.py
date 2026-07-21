import importlib
import json
import sys
import types


class EncodedEmbedding:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeEncoder:
    def encode(self, text):
        return EncodedEmbedding([float(len(text) or 1)])


class FakeRedisClient:
    def __init__(self, *args, **kwargs):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value, ex=None):
        self.cache[key] = value


class FakeIndices:
    def __init__(self):
        self.created = False
        self.created_body = None

    def exists(self, index):
        return self.created

    def create(self, index, body):
        self.created = True
        self.created_body = body


class FakeElasticsearchClient:
    def __init__(self, *args, **kwargs):
        self.indices = FakeIndices()
        self.indexed = []
        self.search_hits = []

    def index(self, index, id, body):
        self.indexed.append({"index": index, "id": id, "body": body})

    def search(self, index, body):
        return {"hits": {"hits": self.search_hits}}


class FakeCollection:
    def __init__(self):
        self.add_calls = []
        self.query_metadatas = []

    def add(self, ids, documents, embeddings, metadatas):
        self.add_calls.append({
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        })

    def query(self, query_embeddings, n_results, include):
        return {
            "metadatas": [self.query_metadatas],
            "documents": [[]],
            "distances": [[]],
        }


class FakeChromaClient:
    def __init__(self, *args, **kwargs):
        self.collection = FakeCollection()

    def get_or_create_collection(self, name):
        return self.collection


def load_storage(monkeypatch, tmp_path, dataset):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    monkeypatch.setenv("DATASET_PATH", str(dataset_path))
    monkeypatch.setenv("DENSE_ENCODER_PATH", "fake-encoder")
    monkeypatch.setenv("ES_HOST", "http://localhost:9200")
    monkeypatch.setenv("ES_INDEX", "answerai-test")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.gguf"))
    monkeypatch.setenv("UVICORN_HOST", "127.0.0.1")
    monkeypatch.setenv("UVICORN_PORT", "8000")

    fake_redis = types.ModuleType("redis")
    fake_redis.Redis = FakeRedisClient
    monkeypatch.setitem(sys.modules, "redis", fake_redis)

    fake_elasticsearch = types.ModuleType("elasticsearch")
    fake_elasticsearch.Elasticsearch = FakeElasticsearchClient
    monkeypatch.setitem(sys.modules, "elasticsearch", fake_elasticsearch)

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.Client = FakeChromaClient
    fake_chroma_config = types.ModuleType("chromadb.config")
    fake_chroma_config.Settings = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setitem(sys.modules, "chromadb.config", fake_chroma_config)

    fake_models = types.ModuleType("app.models")
    fake_models.dense_encoder = FakeEncoder()
    monkeypatch.setitem(sys.modules, "app.models", fake_models)

    for module_name in ("app.storage", "app.utils", "app.config"):
        sys.modules.pop(module_name, None)

    return importlib.import_module("app.storage")


def test_hybrid_retrieve_uses_document_ids_without_list_offsets(monkeypatch, tmp_path):
    storage = load_storage(
        monkeypatch,
        tmp_path,
        [
            {"id": "10", "text": "alpha text"},
            {"id": "20", "text": "beta text"},
            {"id": "30", "text": "gamma text"},
        ],
    )

    storage.es.search_hits = [{"_id": "20", "_source": {"doc_id": "20"}}]
    storage.chroma_collection.query_metadatas = [{"id": "10"}, {"id": "30"}]

    docs, _ = storage.hybrid_retrieve("question", top_k=3, bm25_top=3)

    assert docs == ["beta text", "alpha text", "gamma text"]


def test_indexes_canonical_ids_and_text_for_both_backends(monkeypatch, tmp_path):
    storage = load_storage(
        monkeypatch,
        tmp_path,
        [
            {"id": "doc-a", "text": "alpha text"},
            {"id": "doc-b", "text": "beta text"},
        ],
    )

    assert storage.es.indices.created_body == {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "text": {"type": "text"},
            }
        }
    }
    assert storage.es.indexed == [
        {
            "index": "answerai-test",
            "id": "doc-a",
            "body": {"doc_id": "doc-a", "text": "alpha text"},
        },
        {
            "index": "answerai-test",
            "id": "doc-b",
            "body": {"doc_id": "doc-b", "text": "beta text"},
        },
    ]

    assert storage.chroma_collection.add_calls == [
        {
            "ids": ["doc-a", "doc-b"],
            "documents": ["alpha text", "beta text"],
            "embeddings": [[10.0], [9.0]],
            "metadatas": [{"id": "doc-a"}, {"id": "doc-b"}],
        }
    ]


def test_rank_merge_deduplicates_without_changing_order(monkeypatch, tmp_path):
    storage = load_storage(
        monkeypatch,
        tmp_path,
        [
            {"id": "a", "text": "alpha"},
            {"id": "b", "text": "beta"},
            {"id": "c", "text": "gamma"},
        ],
    )

    storage.es.search_hits = [
        {"_id": "b", "_source": {"doc_id": "b"}},
        {"_id": "c", "_source": {"doc_id": "c"}},
    ]
    storage.chroma_collection.query_metadatas = [{"id": "c"}, {"id": "a"}]

    docs, _ = storage.hybrid_retrieve("question", top_k=3, bm25_top=3)

    assert docs == ["beta", "gamma", "alpha"]


def test_empty_dataset_does_not_index_chroma_and_retrieves_empty_context(monkeypatch, tmp_path):
    storage = load_storage(monkeypatch, tmp_path, [])

    docs, _ = storage.hybrid_retrieve("question", top_k=3, bm25_top=3)

    assert storage.chroma_collection.add_calls == []
    assert docs == []
