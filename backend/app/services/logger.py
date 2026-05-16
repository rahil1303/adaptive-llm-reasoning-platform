"""
Interaction logger: stores all queries, responses, critiques, and metadata
in JSONL format for later analysis.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(
    session_id: str,
    event_type: str,
    data: dict,
) -> str:
    """
    Log an interaction event.
    event_type: query | response | critique | insight | error
    Returns the event ID.
    """
    event_id = str(uuid.uuid4())[:8]
    entry = {
        "event_id": event_id,
        "session_id": session_id,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }

    log_file = LOG_DIR / f"{session_id}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return event_id


def get_session_log(session_id: str) -> list[dict]:
    """Read all events for a session."""
    log_file = LOG_DIR / f"{session_id}.jsonl"
    if not log_file.exists():
        return []

    events = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def new_session_id() -> str:
    return str(uuid.uuid4())[:12]
