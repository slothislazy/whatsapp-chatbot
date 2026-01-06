import json
import logging
import re
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTACT_STORE_PATH = PROJECT_ROOT / "threads_db_store" / "contacts.json"

WIB = timezone(timedelta(hours=7))

DEFAULT_UNKNOWN_CATEGORY = "external_unknown"
_STORE_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(WIB).isoformat(timespec="seconds")


def _split_contact_id(raw: str) -> Tuple[str, str]:
    value = (raw or "").strip()
    if not value:
        return "", ""
    lowered = value.lower()
    if lowered.startswith("lid:"):
        return value[4:], "lid"
    if "@" in value:
        base, suffix = value.split("@", 1)
        return base, suffix.lower()
    return value, ""


def _sanitize_contact_base(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value or "")


def normalize_phone(raw: str) -> str:
    """
    Normalize a WhatsApp identifier for contact lookup.
    Supports digits, alphanumeric LID values, and lid: prefixes.
    Returns empty string when the input cannot be normalized (e.g., group ids).
    """
    base, suffix = _split_contact_id(raw)
    if not base:
        return ""
    if "-" in base:
        return ""
    cleaned = _sanitize_contact_base(base)
    if not cleaned:
        return ""
    if suffix == "lid":
        return f"lid:{cleaned}"
    if re.search(r"[A-Za-z]", cleaned):
        return cleaned
    return cleaned


def _candidate_contact_keys(raw: str) -> list[str]:
    base, suffix = _split_contact_id(raw)
    if not base:
        return []
    if "-" in base:
        return []
    cleaned = _sanitize_contact_base(base)
    if not cleaned:
        return []

    lid_key = f"lid:{cleaned}"
    keys: list[str] = []
    prefer_lid = suffix == "lid"

    if prefer_lid:
        keys.append(lid_key)

    keys.append(cleaned)

    if not prefer_lid:
        keys.append(lid_key)

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _normalize_contact(payload: Dict[str, Any], *, key: str | None = None) -> Dict[str, Any]:
    now = _now_iso()
    raw_phone = payload.get("phone") or key or ""
    normalized_phone = normalize_phone(raw_phone) or str(raw_phone)
    if key:
        normalized_phone = key
    contact = {
        "phone": normalized_phone,
        "category": str(payload.get("category") or DEFAULT_UNKNOWN_CATEGORY),
        "allow_bot": bool(payload.get("allow_bot", True)),
        "created_at": str(payload.get("created_at") or now),
        "updated_at": str(payload.get("updated_at") or now),
        "source": str(payload.get("source") or "unknown"),
        "routing_reason": str(payload.get("routing_reason") or ""),
        "routing_model": str(payload.get("routing_model") or ""),
    }
    return contact


def _ensure_store_dir() -> None:
    CONTACT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_store() -> Dict[str, Dict[str, Any]]:
    if not CONTACT_STORE_PATH.exists():
        return {}
    try:
        raw = json.loads(CONTACT_STORE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                phone = normalize_phone(key or value.get("phone"))
            else:
                phone = normalize_phone(key)
            if not phone:
                continue
            payload = value if isinstance(value, dict) else {"phone": phone}
            normalized[phone] = _normalize_contact(payload, key=phone)
        return normalized
    except Exception:
        logging.exception("Failed to read contact store from %s", CONTACT_STORE_PATH)
        return {}


def _save_store(store: Dict[str, Dict[str, Any]]) -> None:
    _ensure_store_dir()
    payload = json.dumps(store, indent=2)
    tmp_path = CONTACT_STORE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    try:
        tmp_path.replace(CONTACT_STORE_PATH)
    except Exception:
        logging.exception("Failed to atomically replace contact store; attempting fallback write.")
        CONTACT_STORE_PATH.write_text(payload, encoding="utf-8")


def get_contact(phone: str) -> Dict[str, Any] | None:
    candidates = _candidate_contact_keys(phone)
    if not candidates:
        return None
    with _STORE_LOCK:
        store = _load_store()
        for key in candidates:
            contact = store.get(key)
            if contact:
                return _normalize_contact(contact, key=key)
        return None


def ensure_contact_record(
    phone: str,
    *,
    default_category: str = DEFAULT_UNKNOWN_CATEGORY,
    default_allow_bot: bool = True,
    source: str = "auto",
) -> Tuple[Dict[str, Any] | None, bool]:
    """
    Ensure a contact record exists. Returns (contact, created_flag).
    """
    candidates = _candidate_contact_keys(phone)
    if not candidates:
        return None, False

    with _STORE_LOCK:
        store = _load_store()
        contact = None
        normalized = ""
        for key in candidates:
            contact = store.get(key)
            if contact:
                normalized = key
                break
        created = False
        if contact is None:
            created = True
            now = _now_iso()
            normalized = normalize_phone(phone)
            if not normalized:
                return None, False
            contact = {
                "phone": normalized,
                "category": default_category,
                "allow_bot": bool(default_allow_bot),
                "created_at": now,
                "updated_at": now,
                "source": source,
                "routing_reason": "",
                "routing_model": "",
            }
        contact = _normalize_contact(contact, key=normalized)
        store[normalized] = contact
        _save_store(store)
        return contact, created


def upsert_contact(
    phone: str,
    *,
    category: str | None = None,
    allow_bot: bool | None = None,
    source: str | None = None,
    routing_reason: str | None = None,
    routing_model: str | None = None,
) -> Dict[str, Any] | None:
    candidates = _candidate_contact_keys(phone)
    if not candidates:
        return None

    with _STORE_LOCK:
        store = _load_store()
        normalized = ""
        contact = None
        for key in candidates:
            contact = store.get(key)
            if contact:
                normalized = key
                break
        if contact is None:
            normalized = normalize_phone(phone)
            if not normalized:
                return None
            contact = {"phone": normalized}
        contact = _normalize_contact(contact, key=normalized)

        if category is not None:
            contact["category"] = str(category or DEFAULT_UNKNOWN_CATEGORY)
        if allow_bot is not None:
            contact["allow_bot"] = bool(allow_bot)
        if source:
            contact["source"] = str(source)
        if routing_reason is not None:
            contact["routing_reason"] = str(routing_reason)
        if routing_model is not None:
            contact["routing_model"] = str(routing_model)

        now = _now_iso()
        contact.setdefault("created_at", now)
        contact["updated_at"] = now

        store[normalized] = contact
        _save_store(store)
        return contact
