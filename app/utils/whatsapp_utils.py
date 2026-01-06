import json
import logging
import os
import random
import re
import time
from collections import OrderedDict
from textwrap import dedent
from threading import Lock, Timer
from typing import Any, Dict

import ollama
import requests
from flask import current_app, jsonify

from app.services.contact_store import (
    DEFAULT_UNKNOWN_CATEGORY,
    ensure_contact_record,
    get_contact,
    upsert_contact,
)
from app.services.rag_ollama_whatsapp import (
    DEFAULT_GEN_MODEL,
    generate_response,
    pause_thread_for_manual_message,
)

ROUTING_CLASSIFIER_PROMPT = dedent(
    """
    Kamu adalah sistem klasifikasi pesan WhatsApp untuk nomor pribadi.
    Tujuan utama: chatbot HANYA boleh menjawab PESAN PEMBUKA (start of chat)
    yang benar-benar merupakan inquiry baru (bisnis atau akademik).
    Semua pesan lain WAJIB tidak dijawab bot.

    Balas HANYA dengan satu baris JSON valid, TANPA teks tambahan:
    {"category":"customer_question|academic_help|internal_or_partner|other","allow_bot":true|false,"reason":"alasan singkat"}

    =========================
    ATURAN UTAMA (WAJIB TAAT)
    =========================
    - Tidak semua pertanyaan adalah inquiry.
    - Pertanyaan sosial, meta, atau relasional BUKAN inquiry.
    - Default aman: jika ragu/ambigu/samar/vague → category="other", allow_bot=false.
    - Bot TIDAK boleh menjawab pesan lanjutan, operasional, atau sosial.
    - Jika bentuk pesan seperti iklan/promosi → category="other", allow_bot=false.
    - Jika pesan tidak masuk akal → category="other", allow_bot=false.
    - Jika konteks tidak jelas / terlalu pendek / tidak ada tujuan yang bisa dipastikan → category="other", allow_bot=false.

    =========================
    LANGKAH KEPUTUSAN
    =========================

    STEP 0 — HARD OVERRIDE: Nama khusus (langsung stop).
    Jika pesan menyebut "Eddy Rusly" (case-insensitive), termasuk variasi seperti:
    - "Pak Eddy Rusly"
    - "Eddy Rusly"
    - "Pak Eddy"
    MAKA:
    - category="internal_or_partner"
    - allow_bot=false
    - reason="Menyebut nama partner/vendor (Eddy Rusly)"
    DAN JANGAN lanjut ke step berikutnya.

    STEP 1 — Deteksi pesan lanjutan (ongoing conversation).
    Jika salah satu indikator muncul, anggap lanjutan → category="internal_or_partner", allow_bot=false.
    Indikator:
    - Jawaban pendek/konfirmasi: "iya", "ok", "sip", "boleh", "ga jadi", "jadi", "udah", "nanti", "besok", "tadi"
    - Rujukan konteks sebelumnya: "yang kemarin", "seperti tadi", "soal itu", "yang itu", "lanjutin"
    - Tindak lanjut: "aku udah kirim", "sudah saya transfer", "sudah saya isi", "cek ya"
    - Kata sambung lanjutan: "btw", "oh ya", "jadi", "nah", "terus", "kalo gitu"
    - Pesan sangat pendek / fragmen / angka / emoji

    STEP 2 — Deteksi pesan operasional partner/vendor/internal.
    Jika pesan bersifat INFORMASI, KOORDINASI, LOGISTIK, atau UPDATE → category="internal_or_partner", allow_bot=false.
    Indikator kuat (cukup 1):
    - Instruksi/SOP: "sekedar informasi", "mohon", "harap", "untuk jadwal", "hubungi kembali",
      "hari kerja", "senin-kamis", "jumat", "maksimal 2 hari sebelumnya"
    - Jadwal operasional/pickup/pengiriman/pengambilan dokumen
    - Logistik/akses lokasi: "susah masuk", "akses", "truck", "jalan", "site", "loading", "gudang"
    - Update spesifikasi/kapasitas: "unit", "kapasitas", "ton/hour", "mt", "silo", "tambah 1 unit"
    - Nada dominan berupa update, bukan permintaan layanan

    STEP 3 — Deteksi pertanyaan sosial / meta (BUKAN inquiry).
    Jika pesan berupa pertanyaan ringan yang bersifat sosial, relasional, atau meta,
    dan TIDAK bertujuan meminta layanan, bantuan akademik, atau solusi profesional,
    MAKA → category="other", allow_bot=false.
    Contoh kuat (cukup 1):
    - Status kontak/relasi:
      "nomorku kesimpen gak?", "ini siapa?", "aku chat siapa?",
      "dapet nomorku dari mana?", "ini nomor kantor atau pribadi?"
    - Availability ringan:
      "lagi sibuk?", "online?", "bisa telpon?", "sempat?"
    - Small talk tanpa konteks layanan:
      "apa kabar?", "lagi apa?", "di mana?"
    - Pertanyaan YA/TIDAK pendek tanpa konteks layanan

    STEP 4 — Deteksi VAGUE / AMBIGU / SULIT DITENTUKAN (FORCE OTHER).
    Jika pesan:
    - terlalu pendek dan tanpa konteks layanan/akademik yang jelas, ATAU
    - hanya 1 kalimat umum tanpa objek/tujuan (mis: "bisa?", "maksudnya gimana?", "tolong", "urgent", "butuh bantuan"), ATAU
    - tidak menyebut topik (produk/layanan/akademik) yang spesifik,
    MAKA → category="other", allow_bot=false, reason="Pesan vague/ambigu (tujuan tidak jelas)"
    DAN STOP.

    STEP 5 — Tentukan apakah ini benar-benar PESAN PEMBUKA.
    Pesan dianggap PESAN PEMBUKA hanya jika mengandung minimal satu OPENING SIGNAL:
    - Salam/perkenalan:
      "halo", "hai", "selamat", "pagi/siang/sore/malam",
      "perkenalkan", "saya ... dari ..."
    - Konteks kontak pertama:
      "Optimaxx", "mau tanya layanan", "baru dapat kontak", "referensi dari"
    - Permintaan eksplisit awal:
      "minta penawaran", "quotation", "proposal", "harga", "biaya",
      "jadwalkan meeting", "demo", "konsultasi", "survey", "kunjungan",
      "butuh vendor", "bisa bantu?"
    - Pertanyaan panjang dengan konteks jelas (bukan operasional 1 kalimat)

    Jika TIDAK ada opening signal → anggap tidak jelas sebagai pembuka
    → category="other", allow_bot=false, reason="Tidak ada opening signal (bukan pembuka / tidak jelas)".

    STEP 6 — Klasifikasi akhir (HANYA jika lolos sebagai PESAN PEMBUKA):
    - academic_help:
      Jika ada konteks akademik:
      "tugas", "kuliah", "skripsi", "tesis", "jurnal", "paper",
      "referensi", "sitasi", "metodologi", "bab",
      "penelitian", "kampus", "dosen", "mahasiswa"
    - customer_question:
      Jika inquiry terkait produk, layanan, harga, demo,
      konsultasi, dukungan teknis, atau meeting bisnis

    STEP 7 — Fallback FINAL:
    Jika masih ada keraguan/ambigu → category="other", allow_bot=false.

    =========================
    ATURAN allow_bot
    =========================
    - allow_bot=true HANYA untuk:
      customer_question atau academic_help
      DAN hanya jika pesan adalah PESAN PEMBUKA yang JELAS.
    - allow_bot=false untuk semua kondisi lain.
    - Jika ragu → category="other", allow_bot=false.

    Jangan menjawab dengan apa pun selain JSON.
    """
).strip()

_CHAT_ID_CACHE: Dict[str, str] = {}
_DEBOUNCE_LOCK = Lock()
_DEBOUNCE_BUFFERS: Dict[str, Dict[str, Any]] = {}
_DEFAULT_DEBOUNCE_SECONDS = 60.0
_MAX_BUFFERED_MESSAGES = 10
_DEFAULT_CLASSIFIER_MODEL = "deepseek-r1:latest"
_HISTORY_CACHE: Dict[str, bool] = {}
_SEEN_MESSAGE_IDS: "OrderedDict[str, float]" = OrderedDict()
_SEEN_MESSAGE_IDS_LOCK = Lock()
_MAX_SEEN_MESSAGE_IDS = 2000
_SEEN_MESSAGE_TTL_SECONDS = 300.0


def _is_duplicate_message_id(message_id: str) -> bool:
    if not message_id:
        return False
    now = time.time()
    cutoff = now - _SEEN_MESSAGE_TTL_SECONDS
    with _SEEN_MESSAGE_IDS_LOCK:
        while _SEEN_MESSAGE_IDS:
            first_id, ts = next(iter(_SEEN_MESSAGE_IDS.items()))
            if ts >= cutoff:
                break
            _SEEN_MESSAGE_IDS.popitem(last=False)
        if message_id in _SEEN_MESSAGE_IDS:
            _SEEN_MESSAGE_IDS.move_to_end(message_id)
            _SEEN_MESSAGE_IDS[message_id] = now
            return True
        _SEEN_MESSAGE_IDS[message_id] = now
        if len(_SEEN_MESSAGE_IDS) > _MAX_SEEN_MESSAGE_IDS:
            _SEEN_MESSAGE_IDS.popitem(last=False)
    return False

_CLARIFYING_PROMPT = (
    "Halo 👋\nAnda telah terhubung dengan chatbot layanan pelanggan Optimaxx.\nAda yang bisa kami bantu terkait layanan atau pertanyaan Anda?"
)
_VAGUE_GREETING_TOKENS = {
    "halo",
    "hai",
    "hi",
    "hello",
    "pagi",
    "siang",
    "sore",
    "malam",
    "selamat",
    "assalamualaikum",
    "salam",
    "permisi",
    "test",
}
_VAGUE_HONORIFICS = {
    "pak",
    "bu",
    "ibu",
    "bapak",
    "mbak",
    "mas",
    "kak",
    "kakak",
    "admin",
    "min",
    "gan",
    "bang",
    "bro",
    "sis",
    "om",
    "tante",
}

from textwrap import dedent


def log_http_response(response):
    logging.info(
        "Status: %s | Content-type: %s",
        response.status_code,
        response.headers.get("content-type"),
    )
    logging.debug("Body: %s", response.text)


def _get_max_message_age_seconds() -> int:
    raw = _get_config_value("WHATSAPP_MAX_MESSAGE_AGE_SECONDS")
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError, AttributeError):
        return 0


def _get_debounce_window_seconds() -> float:
    raw = _get_config_value("WHATSAPP_DEBOUNCE_SECONDS")
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError, AttributeError):
        return _DEFAULT_DEBOUNCE_SECONDS


def _get_classifier_model() -> str:
    configured = _get_config_value("WHATSAPP_CLASSIFIER_MODEL")
    if configured:
        return str(configured).strip()
    try:
        return DEFAULT_GEN_MODEL
    except Exception:
        return _DEFAULT_CLASSIFIER_MODEL


def _is_vague_greeting(message_body: str) -> bool:
    """
    Return True when the message looks like a simple greeting with no intent.
    """
    text = (message_body or "").strip().lower()
    if not text:
        return False
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return True
    remaining = [
        token
        for token in tokens
        if token not in _VAGUE_GREETING_TOKENS and token not in _VAGUE_HONORIFICS
    ]
    return not remaining


def _is_vague_routing_reason(reason: str) -> bool:
    """
    Return True when the classifier reason indicates ambiguity or vagueness.
    """
    text = (reason or "").strip().lower()
    if not text:
        return False
    vague_tokens = (
        "vague",
        "ambigu",
        "ambiguous",
        "tidak jelas",
        "kurang jelas",
        "bukan pembuka",
        "opening signal",
        "ragu",
        "unclear",
        "tidak masuk akal"
    )
    return any(token in text for token in vague_tokens)


def _extract_unix_timestamp(payload: Dict[str, Any]) -> float | None:
    ts_raw = payload.get("timestamp")
    if ts_raw is None:
        return None
    try:
        ts_val = float(ts_raw)
    except (TypeError, ValueError):
        return None
    if ts_val > 1e12:
        ts_val /= 1000.0
    if ts_val > 1e10:
        ts_val /= 1000.0
    return ts_val


def _is_stale_message(payload: Dict[str, Any], max_age_seconds: int) -> bool:
    """
    Return True when the message timestamp is older than the allowed window.
    """
    if max_age_seconds <= 0:
        return False
    ts_val = _extract_unix_timestamp(payload)
    if ts_val is None:
        return False
    age_seconds = time.time() - ts_val
    if age_seconds <= 0:
        return False
    if age_seconds > max_age_seconds:
        logging.info(
            "Skipping stale message (age %.1fs, limit %ss).", age_seconds, max_age_seconds
        )
        return True
    return False


def _get_config_value(key: str, default: str | None = None) -> str | None:
    value = current_app.config.get(key)
    if value is None:
        value = os.getenv(key)
    return (value or default)


def _get_waha_base_url() -> str:
    return (_get_config_value("WAHA_BASE_URL", "http://localhost:3000") or "").rstrip("/")


def _get_waha_session() -> str:
    return _get_config_value("WAHA_SESSION", "default") or "default"


def _build_waha_headers() -> Dict[str, str]:
    headers = {"Content-type": "application/json"}
    api_key = _get_config_value("WAHA_API_KEY")
    if api_key:
        headers["X-Api-Key"] = api_key
    return headers


def _extract_json_object(raw: str) -> Dict[str, Any] | None:
    """
    Parse a JSON object from raw LLM output.
    """
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _normalize_routing_category(raw: str) -> tuple[str, bool]:
    """
    Normalize classifier category into (contact_category, allow_bot_default).
    """
    token = re.sub(r"[^a-z]", "", (raw or "").lower())
    mapping: Dict[str, tuple[str, bool]] = {
        "customerquestion": ("lead", True),
        "customer": ("lead", True),
        "lead": ("lead", True),
        "prospect": ("lead", True),
        "sales": ("lead", True),
        "academichelp": ("academic", True),
        "academic": ("academic", True),
        "student": ("academic", True),
        "internalorpartner": ("vendor", False),
        "internal": ("vendor", False),
        "partner": ("vendor", False),
        "vendor": ("vendor", False),
        "supplier": ("vendor", False),
        "employee": ("vendor", False),
        "personalchat": ("other", False),
        "other": ("other", False),
    }
    return mapping.get(token, ("other", False))


def normalize_wa_id(raw: str) -> str:
    """
    Normalize WhatsApp id for storage (strip domain if present, keep digits/hyphen).
    """
    value = (raw or "").strip()
    if not value:
        return ""
    if "@" in value:
        return value.split("@", 1)[0]
    return value


def normalize_chat_id(raw: str) -> str:
    """
    Convert user-provided ids/numbers to WAHA chatId format.
    - Keep existing ids with '@' untouched.
    - Numbers become '<number>@c.us'.
    - Strings containing '-' (typical group id) become '<id>@g.us'.
    """
    chat_id = (raw or "").strip()
    if not chat_id:
        return ""
    if "@" in chat_id:
        base = chat_id.split("@", 1)[0]
        if base:
            _CHAT_ID_CACHE.setdefault(base, chat_id)
        return chat_id
    if "-" in chat_id:
        return f"{chat_id}@g.us"
    digits = re.sub(r"\D", "", chat_id)
    base = digits or chat_id
    return f"{base}@c.us"


def _log_history_messages(chat_id: str, messages: list[dict]) -> None:
    """
    Log a brief summary of recent messages for visibility when history is detected.
    """
    if not messages:
        return
    try:
        summaries = []
        for msg in messages[:5]:
            if not isinstance(msg, dict):
                continue
            body = str(msg.get("body") or "").replace("\n", " ")
            if len(body) > 200:
                body = body[:200] + "…"
            summaries.append(
                {
                    "id": msg.get("id"),
                    "from": msg.get("from"),
                    "fromMe": bool(msg.get("fromMe")),
                    "timestamp": msg.get("timestamp"),
                    "hasMedia": bool(msg.get("hasMedia")),
                    "ack": msg.get("ack"),
                    "body": body,
                }
            )
        if summaries:
            logging.info(
                "Existing chat history for %s (showing up to %s messages): %s",
                chat_id,
                len(summaries),
                summaries,
            )
    except Exception as exc:
        logging.debug("Failed to log history messages for %s: %s", chat_id, exc)


def _fetch_history_sample(chat_id: str, message_ts: float | None, from_me: bool) -> dict | None:
    """
    Fetch at most one message from WAHA for this chat in the given direction.
    Applies timestamp filter when provided to avoid counting the current message.
    """
    url = f"{_get_waha_base_url()}/api/{_get_waha_session()}/chats/{chat_id}/messages"
    headers = _build_waha_headers()
    params = {
        "limit": 1,
        "downloadMedia": "false",
        "filter.fromMe": "true" if from_me else "false",
    }
    if message_ts is not None:
        params["filter.timestamp.lte"] = int(max(0, message_ts - 1))

    resp = requests.get(url, params=params, headers=headers, timeout=5)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, list) and payload:
        msg = payload[0]
        if isinstance(msg, dict):
            body = str(msg.get("body") or "").strip()
            has_media = bool(msg.get("hasMedia"))
            if not body and not has_media:
                return None
            return msg
    return None


def _has_existing_history(chat_id: str, message_ts: float | None = None) -> bool:
    """
    Check WAHA for prior messages in both directions (from user and from us).
    Returns True only if there is at least one inbound and one outbound message
    older than the current message_ts (when provided). Caches results when no
    timestamp filter is applied.
    """
    normalized = normalize_chat_id(chat_id)
    if not normalized:
        return False

    cached = _HISTORY_CACHE.get(normalized)
    if cached is not None and message_ts is None:
        return cached

    inbound_msg = None
    outbound_msg = None
    try:
        inbound_msg = _fetch_history_sample(normalized, message_ts, from_me=False)
    except Exception as exc:
        logging.debug("Inbound history check failed for %s: %s", normalized, exc)
    try:
        outbound_msg = _fetch_history_sample(normalized, message_ts, from_me=True)
    except Exception as exc:
        logging.debug("Outbound history check failed for %s: %s", normalized, exc)

    history_messages = []
    if inbound_msg:
        history_messages.append(inbound_msg)
    if outbound_msg:
        history_messages.append(outbound_msg)

    has_history = bool(inbound_msg or outbound_msg)
    if has_history:
        _log_history_messages(normalized, history_messages)

    if message_ts is None:
        _HISTORY_CACHE[normalized] = has_history
    return has_history


def _get_test_whitelist() -> set[str]:
    """
    Return a set of allowed WA IDs (digits only) for test runs.
    Empty set means allow all senders.
    """
    raw = _get_config_value("WHATSAPP_TEST_NUMBERS")
    if not raw:
        return set()
    allowed: set[str] = set()
    for token in re.split(r"[,\s]+", str(raw)):
        wa = normalize_wa_id(token)
        if wa:
            allowed.add(wa)
    return allowed


def _is_blocked_by_test_whitelist(wa_id: str) -> bool:
    allowed = _get_test_whitelist()
    if not allowed:
        return False
    normalized = normalize_wa_id(wa_id)
    return not normalized or normalized not in allowed


def _lookup_chat_id_from_waha(base_id: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None

    url = f"{_get_waha_base_url()}/api/{_get_waha_session()}/chats/overview"
    headers = _build_waha_headers()
    try:
        response = requests.get(
            url,
            params={"ids": candidates, "limit": len(candidates)},
            headers=headers,
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            for candidate in candidates:
                if any(isinstance(item, dict) and item.get("id") == candidate for item in payload):
                    return candidate
            for item in payload:
                if isinstance(item, dict) and item.get("id"):
                    return str(item["id"])
    except Exception as exc:
        logging.debug("WAHA chat lookup failed for %s: %s", base_id, exc)
    return None


def _resolve_chat_id(recipient: str) -> str:
    """
    Prefer an existing WAHA chat id (supports @lid/@c.us/@g.us) when only a bare number is provided.
    Falls back to the legacy normalize_chat_id behavior.
    """
    chat_id = (recipient or "").strip()
    if not chat_id:
        return ""
    if "@" in chat_id:
        return normalize_chat_id(chat_id)

    base_id = normalize_wa_id(chat_id)
    if not base_id:
        return normalize_chat_id(chat_id)

    cached = _CHAT_ID_CACHE.get(base_id)
    if cached:
        return cached

    if "-" in base_id:
        candidates = [f"{base_id}@g.us"]
    else:
        digits = re.sub(r"\D", "", base_id)
        base = digits or base_id
        candidates = [f"{base}@c.us", f"{base}@lid"]

    resolved = _lookup_chat_id_from_waha(base_id, candidates)
    if resolved:
        _CHAT_ID_CACHE[base_id] = resolved
        return resolved

    fallback = candidates[0] if candidates else normalize_chat_id(chat_id)
    _CHAT_ID_CACHE[base_id] = fallback
    return fallback


def _debounce_key(wa_id: str, chat_id: str) -> str:
    return (wa_id or "").strip() or (chat_id or "").strip()


def _cancel_debounce_buffer(key: str | None) -> None:
    if not key:
        return
    with _DEBOUNCE_LOCK:
        state = _DEBOUNCE_BUFFERS.pop(key, None)
    if not state:
        return
    timer = state.get("timer")
    if isinstance(timer, Timer):
        try:
            timer.cancel()
        except Exception:
            logging.debug("Failed to cancel debounce timer for %s", key)


def _flush_debounced_messages(key: str, app) -> None:
    with _DEBOUNCE_LOCK:
        state = _DEBOUNCE_BUFFERS.pop(key, None)
    if not state:
        return

    messages = [str(m).strip() for m in state.get("messages") or [] if str(m).strip()]
    if not messages:
        return

    chat_id = state.get("chat_id") or ""
    wa_id = state.get("wa_id") or chat_id
    message_ids = state.get("message_ids") or []
    message_id = message_ids[-1] if message_ids else None
    debounce_seconds = _DEFAULT_DEBOUNCE_SECONDS
    combined_body = "\n".join(messages).strip()
    if not combined_body:
        return

    contact = state.get("contact")
    needs_classification = bool(state.get("needs_classification"))
    routing_decision: Dict[str, Any] | None = None

    try:
        with app.app_context():
            if needs_classification and _is_vague_greeting(combined_body):
                logging.info(
                    "Sending clarification prompt for %s due to vague greeting.",
                    wa_id or chat_id,
                )
                _send_clarification_prompt(chat_id, wa_id, message_id=message_id)
                return
            if needs_classification:
                routing_decision = _classify_message_for_routing(wa_id, combined_body, contact)
                if _is_vague_routing_reason(routing_decision.get("reason", "")):
                    logging.info(
                        "Routing returned vague/ambiguous for %s; sending clarification prompt.",
                        wa_id or chat_id,
                    )
                    _send_clarification_prompt(chat_id, wa_id, message_id=message_id)
                    return
                contact_id = contact.get("phone") if isinstance(contact, dict) else None
                contact = (
                    _persist_routing_decision(wa_id, routing_decision, contact_id=contact_id)
                    or contact
                )
                if not routing_decision.get("allow_bot", True):
                    logging.info(
                        "Routing blocked auto-reply for %s (category=%s, reason=%s).",
                        wa_id or chat_id,
                        routing_decision.get("category", ""),
                        routing_decision.get("reason", ""),
                    )
                    return
            logging.info(
                "Sending debounced reply for %s with %s message(s) after %.1fs of inactivity.",
                wa_id or chat_id,
                len(messages),
                debounce_seconds,
            )
            _reply_with_llm(chat_id, wa_id, combined_body, message_id=message_id)
    except Exception:
        logging.exception("Failed to deliver debounced reply for %s", wa_id or chat_id)


def _schedule_debounced_reply(
    wa_id: str,
    chat_id: str,
    message_body: str,
    message_id: str | None = None,
    contact_context: Dict[str, Any] | None = None,
) -> None:
    debounce_seconds = _get_debounce_window_seconds()
    if debounce_seconds <= 0:
        _reply_with_llm(chat_id, wa_id, message_body, message_id=message_id)
        return

    try:
        app = current_app._get_current_object()
    except Exception:
        logging.exception("No Flask app context; sending reply immediately.")
        _reply_with_llm(chat_id, wa_id, message_body, message_id=message_id)
        return

    key = _debounce_key(wa_id, chat_id)
    if not key:
        _reply_with_llm(chat_id, wa_id, message_body, message_id=message_id)
        return

    needs_classification = bool(contact_context.get("needs_classification")) if contact_context else False
    contact = contact_context.get("contact") if isinstance(contact_context, dict) else None

    with _DEBOUNCE_LOCK:
        state = _DEBOUNCE_BUFFERS.get(key, {"messages": [], "message_ids": []})
        timer = state.get("timer")
        if isinstance(timer, Timer):
            timer.cancel()

        messages = state.get("messages") or []
        messages.append(message_body.strip())
        state["messages"] = messages[-_MAX_BUFFERED_MESSAGES:]

        if message_id:
            ids = state.get("message_ids") or []
            ids.append(message_id)
            state["message_ids"] = ids[-_MAX_BUFFERED_MESSAGES:]

        state["chat_id"] = chat_id
        state["wa_id"] = wa_id or chat_id
        state["debounce_seconds"] = debounce_seconds
        state["needs_classification"] = needs_classification
        if contact is not None:
            state["contact"] = contact

        timer = Timer(debounce_seconds, _flush_debounced_messages, args=(key, app))
        timer.daemon = True
        state["timer"] = timer

        _DEBOUNCE_BUFFERS[key] = state
        timer.start()

def get_text_message_input(recipient, text) -> Dict[str, Any]:
    chat_id = _resolve_chat_id(recipient)
    session = _get_waha_session()
    return {"session": session, "chatId": chat_id, "text": text or ""}


def get_read_receipt_payload(
    chat_id: str, message_id: str | None = None, participant: str | None = None
) -> Dict[str, Any]:
    return {
        "session": _get_config_value("WAHA_SESSION", "default"),
        "chatId": normalize_chat_id(chat_id),
        "messageId": message_id or "",
        "participant": participant or None,
    }


def _typing_payload(chat_id: str) -> Dict[str, Any]:
    return {
        "session": _get_waha_session(),
        "chatId": _resolve_chat_id(chat_id),
    }


def start_typing(chat_id: str):
    send_message(_typing_payload(chat_id), endpoint="startTyping")


def stop_typing(chat_id: str):
    send_message(_typing_payload(chat_id), endpoint="stopTyping")


def typing_pause(chat_id: str, seconds: float = 10.0):
    """
    Send startTyping, wait for the given duration, then stopTyping to mimic a human delay.
    """
    start_typing(chat_id)
    try:
        time.sleep(max(0.0, seconds))
    finally:
        stop_typing(chat_id)


def send_message(payload: Dict[str, Any], *, endpoint: str = "sendText"):
    """
    Send payload to WAHA.
    - endpoint: WAHA API endpoint name (e.g., sendText, sendSeen).
    Returns requests.Response or Flask response tuple on error.
    """
    base_url = _get_waha_base_url()
    headers = _build_waha_headers()

    endpoint_path = endpoint.lstrip("/")
    url = f"{base_url}/api/{endpoint_path}"

    try:
        response = requests.post(
            url,
            json=payload if isinstance(payload, dict) else {},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return jsonify({"status": "error", "message": "Request timed out"}), 408
    except requests.HTTPError as http_err:
        status_code = (
            http_err.response.status_code if http_err.response is not None else 500
        )
        error_body = http_err.response.text if http_err.response is not None else ""
        logging.error(
            "WAHA request failed with status %s. Response body: %s",
            status_code,
            error_body,
        )
        return (
            jsonify({"status": "error", "message": "Failed to send message"}),
            status_code,
        )
    except (
        requests.RequestException
    ) as e:
        logging.error(f"Request failed due to: {e}")
        return jsonify({"status": "error", "message": "Failed to send message"}), 500
    else:
        log_http_response(response)
    return response


def _simulate_human_typing(chat_id: str, *, min_seconds: float = 5.0, max_seconds: float = 10.0) -> None:
    """
    Send typing indicators with a short randomized delay to mimic a human reply cadence.
    """
    delay = random.uniform(min_seconds, max_seconds)
    start_typing(chat_id)
    try:
        time.sleep(delay)
    finally:
        stop_typing(chat_id)


def _send_clarification_prompt(
    chat_id: str, wa_id: str, *, message_id: str | None = None
) -> None:
    """
    Send a short clarifying prompt when the incoming message is too vague.
    """
    if message_id:
        send_message(get_read_receipt_payload(chat_id, message_id), endpoint="sendSeen")
    payload = get_text_message_input(chat_id, _CLARIFYING_PROMPT)
    _simulate_human_typing(chat_id)
    send_message(payload)


def _reply_with_llm(
    chat_id: str, wa_id: str, message_body: str, *, message_id: str | None = None
) -> None:
    if message_id:
        send_message(get_read_receipt_payload(chat_id, message_id), endpoint="sendSeen")
    response = generate_response(message_body, wa_id, message_id=message_id)
    if not response:
        logging.info(
            "No automated reply generated for %s (paused for human assistance or empty response).",
            wa_id,
        )
        return
    response = process_text_for_whatsapp(response)
    data = get_text_message_input(chat_id, response)
    _simulate_human_typing(chat_id)
    send_message(data)

def _classify_message_for_routing(
    wa_id: str, message_body: str, contact: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Lightweight LLM classifier to decide whether the bot should respond.
    Returns dict with keys: category, allow_bot, reason.
    """
    fallback_category = (contact.get("category") if contact else DEFAULT_UNKNOWN_CATEGORY) or DEFAULT_UNKNOWN_CATEGORY
    fallback_allow = bool(contact.get("allow_bot", True)) if contact else True
    result = {
        "category": fallback_category,
        "allow_bot": fallback_allow,
        "reason": "fallback",
        "model": _get_classifier_model(),
    }

    text = (message_body or "").strip()
    if not text:
        result["reason"] = "empty_message"
        return result

    contact_note = (
        f"kontak terdaftar (kategori={contact.get('category','')}, allow_bot={contact.get('allow_bot', True)})"
        if contact
        else "kontak baru/unknown"
    )
    user_prompt = dedent(
        f"""
        Info kontak: {contact_note}
        Pesan (gabungan):
        ---
        {text}
        ---
        """
    ).strip()

    model = result["model"]
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": ROUTING_CLASSIFIER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.1,
                "num_ctx": 3000,
                "top_p": 0.1,
                "repeat_penalty": 1.05,
            },
        )
        raw = resp.message.content
        logging.debug("Routing classifier raw output for %s: %s", wa_id or "<unknown>", raw)
    except Exception as exc:
        logging.warning("Routing classifier failed for %s: %s", wa_id, exc)
        result["reason"] = "classifier_error"
        return result

    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        logging.info("Routing classifier returned non-dict for %s: %s", wa_id, raw)
        result["reason"] = "unparsed_output"
        return result

    mapped_category, mapped_allow = _normalize_routing_category(parsed.get("category", ""))
    allow_bot = bool(parsed.get("allow_bot", mapped_allow)) and mapped_allow
    reason = str(parsed.get("reason") or "").strip() or "llm_routing"
    category = mapped_category or fallback_category
    result.update(
        {
            "category": category,
            "allow_bot": allow_bot,
            "reason": reason,
            "model": model,
        }
    )
    logging.info(
        "Routing decision for %s: category=%s allow_bot=%s reason=%s",
        wa_id or "<unknown>",
        category,
        allow_bot,
        reason,
    )

    print(result)
    return result


def _persist_routing_decision(
    wa_id: str, routing: Dict[str, Any], *, contact_id: str | None = None
) -> Dict[str, Any] | None:
    target_id = contact_id or wa_id
    try:
        return upsert_contact(
            target_id,
            category=routing.get("category") or DEFAULT_UNKNOWN_CATEGORY,
            allow_bot=bool(routing.get("allow_bot", True)),
            source="llm_router",
            routing_reason=routing.get("reason"),
            routing_model=routing.get("model"),
        )
    except Exception:
        logging.exception("Failed to persist routing decision for %s", target_id)
        return None

def _unique_contact_candidates(*values: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value:
            continue
        raw = str(value).strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return out

def _apply_contact_policy(
    wa_id: str,
    chat_id: str,
    message_body: str,
    message_ts: float | None = None,
    *,
    contact_candidates: list[str] | None = None,
) -> Dict[str, Any] | None:
    """
    Contact gate: skip bot for internal/vendor categories and mark new/unknown contacts
    for automatic LLM-based routing. Returns None when the bot should NOT continue processing.
    """
    if not wa_id and not contact_candidates:
        return {"message_body": message_body, "needs_classification": False}

    candidates = _unique_contact_candidates(*(contact_candidates or []), wa_id)
    contact = None
    created = False
    for candidate in candidates:
        try:
            contact = get_contact(candidate)
        except Exception:
            logging.exception("Contact store lookup failed for %s", candidate)
            contact = None
        if contact:
            break

    if contact is None:
        create_id = wa_id or (candidates[0] if candidates else "")
        if not create_id:
            return {"message_body": message_body, "needs_classification": False}
        try:
            contact, created = ensure_contact_record(create_id)
        except Exception:
            logging.exception("Contact store unavailable; defaulting to bot for %s", create_id)
            return {"message_body": message_body, "needs_classification": False}

    if contact is None:
        return {"message_body": message_body, "needs_classification": False}

    contact_id = contact.get("phone") or wa_id
    is_new_contact = bool(created)

    if not contact.get("allow_bot", True):
        logging.info(
            "Skipping bot for %s because allow_bot is already disabled (category=%s).",
            contact_id or wa_id,
            contact.get("category", ""),
        )
        return None

    if is_new_contact and _has_existing_history(chat_id, message_ts=message_ts):
        upsert_contact(
            contact_id or wa_id,
            category="other",
            allow_bot=False,
            source="history_detect",
            routing_reason="preexisting_chat_history",
            routing_model="waha_history_check",
        )
        logging.info(
            "Detected existing chat history for %s; categorizing as 'other' and disabling bot.",
            contact_id or wa_id,
        )
        return None
    
    src = (contact.get("source") or "").lower()
    needs_classification = False
    if created or contact.get("category") == DEFAULT_UNKNOWN_CATEGORY:
        needs_classification = True
    elif contact.get("allow_bot", True) and src not in {"llm_router", "dashboard_toggle", "handoff_detect", "manual_outbound"}:
        needs_classification = True

    if not contact.get("allow_bot", True) and not needs_classification:
        logging.info(
            "Skipping bot for %s (category=%s, allow_bot=False).",
            contact_id or wa_id,
            contact.get("category", ""),
        )
        return None

    return {
        "message_body": message_body,
        "contact": contact,
        "needs_classification": needs_classification,
    }


def process_whatsapp_message(body):
    if not is_valid_whatsapp_message(body):
        logging.warning("Invalid WhatsApp webhook payload; skipping processing.")
        return

    try:
        waha_payload = extract_message_payload(body)
        if not isinstance(waha_payload, dict):
            logging.warning("No WhatsApp message payload found after normalization.")
            return
        chat_id_raw = str(waha_payload.get("from") or waha_payload.get("author") or "").strip()
        wa_id = normalize_wa_id(chat_id_raw)
        message_body = str(waha_payload.get("body") or "").strip()
        message_id = str(waha_payload.get("id") or "").strip()
        has_media = bool(waha_payload.get("hasMedia"))
        from_me = bool(waha_payload.get("fromMe"))
        source = str(waha_payload.get("source") or "").lower()
        message_ts = _extract_unix_timestamp(waha_payload)
        to_raw = str(waha_payload.get("to") or "").strip()
        lid_raw = str(waha_payload.get("lid") or "").strip()
    except Exception:
        logging.warning("Unable to parse WAHA webhook payload; skipping processing.")
        return

    chat_id = normalize_chat_id(chat_id_raw or wa_id)

    if _is_stale_message(waha_payload, _get_max_message_age_seconds()):
        logging.info(
            "Ignoring old WhatsApp message %s for %s", message_id or "<no-id>", wa_id or chat_id
        )
        return

    if _is_duplicate_message_id(message_id):
        logging.info(
            "Skipping duplicate WhatsApp message %s for %s",
            message_id or "<no-id>",
            wa_id or chat_id,
        )
        return

    if from_me and source == "api":
        logging.info("Skipping API-originated self message for %s", wa_id or chat_id_raw)
        return

    if _is_blocked_by_test_whitelist(wa_id):
        logging.info("Skipping sender %s because they are not in WHATSAPP_TEST_NUMBERS.", wa_id)
        return

    if from_me:
        target_raw = to_raw or chat_id_raw or wa_id
        target_wa_id = normalize_wa_id(target_raw)
        target_chat_id = normalize_chat_id(target_raw or chat_id)
        logging.info(
            "Detected manual outbound message from %s; pausing bot for %s.",
            wa_id or chat_id,
            target_wa_id or target_chat_id,
        )
        _cancel_debounce_buffer(_debounce_key(target_wa_id, target_chat_id))
        pause_thread_for_manual_message(
            target_wa_id or target_chat_id,
            message_body=message_body,
        )
        return

    contact_candidates = _unique_contact_candidates(lid_raw, chat_id_raw, wa_id)
    contact_decision = _apply_contact_policy(
        wa_id,
        chat_id,
        message_body,
        message_ts,
        contact_candidates=contact_candidates,
    )
    if contact_decision is None:
        _cancel_debounce_buffer(_debounce_key(wa_id, chat_id))
        return

    message_body = str(contact_decision.get("message_body") or "").strip()
    logging.info(
        "Processing WhatsApp message %s", message_body,
    )

    if has_media:
        logging.info(
            "Ignoring non-text WhatsApp message with media for %s; no automated reply sent.",
            wa_id or chat_id,
        )
        return

    if not message_body:
        logging.info(
            "Ignoring unsupported/empty WhatsApp message type for %s; no automated reply sent.",
            wa_id or chat_id,
        )
        return

    _schedule_debounced_reply(
        wa_id,
        chat_id,
        message_body,
        message_id=message_id,
        contact_context=contact_decision,
    )


def process_text_for_whatsapp(text):
    if not text:
        return ""
    
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"`{3}.*?`{3}", "", text, flags=re.DOTALL)
    text = text.strip()

    text = text.replace("#", "")
    text = text.replace("(https://optimaxx.id/id)", "")
    text = text.replace("kanal", "channel")

    return re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)


def extract_message_payload(body: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Normalize incoming WAHA webhook payloads into a WAHA-like payload dict.
    Expected keys in the returned dict: from/author, body, id, fromMe, participant, source.
    """
    if not isinstance(body, dict):
        return None

    payload = body.get("payload")
    if isinstance(payload, dict):
        return payload

    return None


def is_valid_whatsapp_message(body):
    """
    Check if the incoming WAHA webhook event has a valid WhatsApp message structure.
    """
    try:
        if not isinstance(body, dict):
            return False
        event = body.get("event")
        if event and event not in {"message", "message.any"}:
            return False

        payload = extract_message_payload(body)
        if not isinstance(payload, dict):
            return False
        if payload.get("fromMe") and str(payload.get("source") or "").lower() == "api":
            return False
        chat_id = payload.get("from") or payload.get("author")
        return bool(chat_id)
    except Exception:
        return False
