
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_LOG = LOG_DIR / "api_requests.jsonl"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log_request(
    endpoint,
    method="POST",
    status_code=200,
    latency_ms=0,
    authenticated=True,
    request_id=None,
    error=None
):

    if request_id is None:
        request_id = str(uuid.uuid4())

    record = {
        "timestamp": utc_now(),
        "request_id": request_id,
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "latency_ms": round(float(latency_ms), 3),
        "authenticated": bool(authenticated),
        "error": error
    }

    with open(REQUEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record
