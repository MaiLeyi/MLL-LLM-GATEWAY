from datetime import datetime, timezone


def utc_now_iso() -> str:
    """统一时间戳格式：带毫秒与 UTC 时区，文本排序即时间排序。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
