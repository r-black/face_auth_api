import os
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType, Collection, utility
)
from app.core.config import settings


def connect():
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )


def ensure_collection():
    name = settings.MILVUS_COLLECTION
    dim = settings.MILVUS_DIM

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",  # используем IP на нормализованных эмбеддингах
        "params": {"nlist": 128},
    }

    if utility.has_collection(name):
        collection = Collection(name)
        # Проверим, есть ли индекс на поле embedding и соответствует ли он desired metric
        need_reindex = True
        try:
            for idx in collection.indexes:
                # idx.params содержит dict с "metric_type", "index_type" и др.
                if idx.field_name == "embedding":
                    existing_metric = idx.params.get("metric_type", "").upper()
                    existing_index_type = idx.params.get("index_type", "").upper()
                    if existing_metric == "IP" and existing_index_type == "IVF_FLAT":
                        need_reindex = False  # уже то, что нужно
                    else:
                        # удалим старый некорректный/неподходящий индекс
                        collection.drop_index("embedding")
                    break
        except Exception:
            # на всякий случай пересоздадим индекс ниже
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


def search_similar(collection: Collection, embedding, top_k=1):
    # На случай, если embedding вдруг не нормализован — нормализуем ещё раз (безопасно)
    import numpy as np

    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    emb = normalize(np.array(embedding))
    results = collection.search(
        data=[emb.tolist()],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
    )
    return results
