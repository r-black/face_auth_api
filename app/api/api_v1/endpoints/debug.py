from collections import Counter

from fastapi import APIRouter
from fastapi import Path

router = APIRouter()


@router.get("/debug/history/{user_id}")
async def debug_user_history(user_id: int = Path(..., description="ID пользователя для истории")):
    """
    Показывает статистику по сохранённым эмбеддингам в истории пользователя.
    """
    try:
        # Получаем коллекцию истории (создаст, если ещё нет)
        from app.services.milvus import init_history_collection  # или поправь путь, если module отличается
        coll = init_history_collection()

        # Выбираем все записи для user_id
        expr = f"user_id == {user_id}"
        # Запросим scalar поля: user_id, created_at, source
        results = coll.query(expr=expr, output_fields=["user_id", "created_at", "source"])

        if not results:
            return {
                "user_id": user_id,
                "total": 0,
                "by_source": {},
                "first_seen": None,
                "last_seen": None,
                "recent_records": [],
                "note": "No embeddings found for this user",
            }

        # Преобразуем и сортируем по created_at
        # each result is dict like {"user_id":..., "created_at":..., "source":...}
        sorted_by_time = sorted(results, key=lambda r: r.get("created_at", 0))
        created_times = [r["created_at"] for r in sorted_by_time if "created_at" in r]
        sources = [r.get("source", "unknown") for r in sorted_by_time]

        # Конвертация timestamp в readable (если нужно)
        from datetime import datetime
        def fmt(ts_ms):
            try:
                return datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + "Z"
            except Exception:
                return None

        first_seen = fmt(created_times[0]) if created_times else None
        last_seen = fmt(created_times[-1]) if created_times else None

        # Интервалы между сохранениями
        intervals = []
        for earlier, later in zip(created_times, created_times[1:]):
            intervals.append(later - earlier)
        avg_interval_ms = sum(intervals) / len(intervals) if intervals else None

        # Соберём последние N записей для вывода
        recent = sorted_by_time[-5:]
        recent_records = [
            {
                "created_at": fmt(r["created_at"]) if "created_at" in r else None,
                "source": r.get("source"),
            }
            for r in reversed(recent)
        ]

        return {
            "user_id": user_id,
            "total": len(results),
            "by_source": dict(Counter(sources)),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "average_interval_ms": avg_interval_ms,
            "recent_records": recent_records,
        }

    except Exception:
        logger.exception("failed to fetch user history for debug")
        raise HTTPException(status_code=500, detail="Internal server error")