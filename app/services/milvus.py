import time

import numpy as np
import logging
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from app.core.config import settings


logger = logging.getLogger(__name__)


HISTORY_COLLECTION = "face_embeddings_history"

_COLLECTION_CACHE: dict[str, Collection] = {}


def get_connection():
    # idempotent: повторный connect ничего страшного не сделает
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )


def _ensure_collection_internal():
    name = settings.MILVUS_COLLECTION
    dim = settings.MILVUS_DIM

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128},
    }

    if utility.has_collection(name):
        collection = Collection(name)
        need_reindex = True
        try:
            for idx in collection.indexes:
                if idx.field_name == "embedding":
                    existing_metric = idx.params.get("metric_type", "").upper()
                    existing_index_type = idx.params.get("index_type", "").upper()
                    if existing_metric == "IP" and existing_index_type == "IVF_FLAT":
                        need_reindex = False
                    else:
                        collection.drop_index("embedding")
                    break
        except Exception:
            need_reindex = True

        if need_reindex:
            collection.create_index("embedding", index_params)
    else:
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
        )
        vector_field = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        )
        schema = CollectionSchema([id_field, vector_field], description="Face embeddings")
        collection = Collection(name, schema=schema)
        collection.create_index("embedding", index_params)

    collection.load()
    return collection


def get_collection():
    if settings.MILVUS_COLLECTION in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[settings.MILVUS_COLLECTION]
    get_connection()
    collection = _ensure_collection_internal()
    _COLLECTION_CACHE[settings.MILVUS_COLLECTION] = collection
    return collection


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def init_history_collection():
    """Инициализирует (если нужно) коллекцию с user_id, created_at и source."""
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )

    dim = settings.MILVUS_DIM
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128},
    }

    if utility.has_collection(HISTORY_COLLECTION):
        collection = Collection(HISTORY_COLLECTION)
    else:
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        user_field = FieldSchema(name="user_id", dtype=DataType.INT64)
        created_at_field = FieldSchema(name="created_at", dtype=DataType.INT64)
        source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64)

        schema = CollectionSchema(
            [id_field, embedding_field, user_field, created_at_field, source_field],
            description="Face embeddings history per user",
        )
        collection = Collection(HISTORY_COLLECTION, schema=schema)
        collection.create_index("embedding", index_params)
    collection.load()
    return collection

def save_user_embedding(user_id: int, embedding: np.ndarray, source: str = "reauth"):
    coll = init_history_collection()
    emb_norm = normalize_embedding(np.array(embedding))
    ts_ms = int(time.time() * 1000)
    # Для auto_id PRIMARY field не передаём список пустой.
    # Порядок: embedding, user_id, created_at, source — 4 списка.
    coll.insert([[emb_norm.tolist()], [user_id], [ts_ms], [source]])
    coll.flush()


def search_user_history(embedding: np.ndarray, user_id: int, top_k: int = 3):
    coll = init_history_collection()
    emb_norm = normalize_embedding(np.array(embedding))
    expr = f"user_id == {user_id}"
    try:
        results = coll.search(
            data=[emb_norm.tolist()],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
        )
        return results
    except Exception as e:
        # если что-то с фильтрацией не так, откатываемся — caller сам решит fallback
        raise
