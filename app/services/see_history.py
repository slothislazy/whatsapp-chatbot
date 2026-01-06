import logging
import shelve
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import (
    Blueprint,
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from app.utils.whatsapp_utils import (
    get_text_message_input,
    send_message,
    normalize_wa_id,
    _simulate_human_typing
)
from app.services import contact_store
from app.services.contact_store import upsert_contact
from app.config import load_configurations, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
DB_BASENAME = PROJECT_ROOT / "threads_db_store" / "threads_db"

history_blueprint = Blueprint(
    "conversation_history", __name__, template_folder=str(TEMPLATES_DIR)
)

DEFAULT_THREAD_STATE = {
    "messages": [],
    "ai_paused": False,
    "handoff_reason": "",
    "handoff_ts": "",
}
MANUAL_MESSAGE_ROLE = "operator"
WIB = timezone(timedelta(hours=7))


def _timestamp() -> str:
    return datetime.now(WIB).isoformat(timespec="seconds")


def _coerce_message(raw) -> dict:
    if isinstance(raw, dict):
        role = (raw.get("role") or "unknown").strip() or "unknown"
        content = str(raw.get("content") or "")
        timestamp = str(raw.get("timestamp") or _timestamp())
        ai_readable = bool(raw.get("ai_readable", True))
    else:
        role = "unknown"
        content = str(raw)
        timestamp = _timestamp()
        ai_readable = True
    return {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "ai_readable": ai_readable,
    }


def _coerce_thread_state(wa_id: str, raw) -> dict:
    state = dict(DEFAULT_THREAD_STATE)
    state["wa_id"] = wa_id
    if isinstance(raw, dict):
        state["ai_paused"] = bool(raw.get("ai_paused", state["ai_paused"]))
        state["handoff_reason"] = str(raw.get("handoff_reason") or "")
        state["handoff_ts"] = str(raw.get("handoff_ts") or "")
        messages = raw.get("messages", [])
    elif isinstance(raw, list):
        messages = raw
    else:
        messages = []
    state["messages"] = [_coerce_message(item) for item in messages]
    return state


@contextmanager
def _open_thread_store(flag: str = "c"):
    """
    Open the threads_db shelf. Rely on the default dbm (gdbm on Debian-based images),
    which keeps a single threads_db file instead of multiple dumb-db artifacts.
    """
    DB_BASENAME.parent.mkdir(parents=True, exist_ok=True)
    db = shelve.open(str(DB_BASENAME), flag=flag)
    try:
        yield db
    finally:
        db.close()


def _load_thread_state(wa_id: str) -> dict:
    with _open_thread_store() as db:
        return _coerce_thread_state(wa_id, db.get(wa_id))


def _save_thread_state(wa_id: str, state: dict) -> None:
    payload = {
        "messages": state.get("messages", []),
        "ai_paused": bool(state.get("ai_paused")),
        "handoff_reason": state.get("handoff_reason") or "",
        "handoff_ts": state.get("handoff_ts") or "",
    }
    with _open_thread_store() as db:
        db[wa_id] = payload


def load_conversations():
    try:
        with _open_thread_store(flag="c") as db:
            conversations = [
                _coerce_thread_state(wa_id, history) for wa_id, history in db.items()
            ]
    except (FileNotFoundError, OSError) as exc:
        logging.warning("Unable to open conversation store %s: %s", DB_BASENAME, exc)
        return []
    conversations.sort(key=lambda item: item["wa_id"])
    return conversations


def _load_contacts_store() -> list[dict]:
    """
    Snapshot contact records from contacts.json for display in the dashboard.
    """
    try:
        # Reuse the contact_store helpers to keep normalization consistent.
        with contact_store._STORE_LOCK:  # noqa: SLF001
            store = contact_store._load_store()  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "Unable to read contact store %s: %s",
            contact_store.CONTACT_STORE_PATH,
            exc,
        )
        return []

    if not isinstance(store, dict):
        return []

    records = []
    for phone, payload in sorted(store.items()):
        if isinstance(payload, dict):
            record = dict(payload)
            record["phone"] = record.get("phone") or phone
            records.append(record)
    return records


@history_blueprint.route("/")
def history_index():
    contacts = load_conversations()
    selected_wa_id = normalize_wa_id(request.args.get("wa_id") or "")
    selected_contact = None

    for contact in contacts:
        messages = contact["messages"]
        last_text = ""
        last_role = ""
        if messages:
            last = messages[-1]
            last_text = (last.get("content") or "").strip()
            last_role = (last.get("role") or "message").capitalize()
        contact["last_text"] = last_text
        contact["last_role"] = last_role
        if contact["wa_id"] == selected_wa_id:
            selected_contact = contact

    if not selected_contact and contacts:
        selected_contact = contacts[0]
        selected_wa_id = selected_contact["wa_id"]

    selected_messages = selected_contact["messages"] if selected_contact else []
    contact_store_records = _load_contacts_store()
    return render_template(
        "history_view.html",
        contacts=contacts,
        selected_contact=selected_contact,
        selected_wa_id=selected_wa_id,
        selected_messages=selected_messages,
        db_path=DB_BASENAME.resolve(),
        contact_store_records=contact_store_records,
        contact_store_path=contact_store.CONTACT_STORE_PATH.resolve(),
        status_message=request.args.get("status"),
        error_message=request.args.get("error"),
    )


@history_blueprint.route("/feed/<wa_id>")
def history_feed(wa_id: str):
    wa_id = normalize_wa_id(wa_id or "")
    if not wa_id:
        return jsonify({"error": "WhatsApp number is required."}), 400
    try:
        state = _load_thread_state(wa_id)
    except OSError as exc:
        logging.warning("Unable to load thread state for %s: %s", wa_id, exc)
        state = _coerce_thread_state(wa_id, [])
    return jsonify(
        {
            "wa_id": wa_id,
            "messages": state["messages"],
            "ai_paused": state["ai_paused"],
            "handoff_reason": state["handoff_reason"],
            "handoff_ts": state["handoff_ts"],
            "message_count": len(state["messages"]),
        }
    )


@history_blueprint.route("/send", methods=["POST"])
def send_manual_message():
    wa_id = normalize_wa_id(request.form.get("wa_id") or "")
    message = (request.form.get("message") or "").strip()

    if not wa_id:
        return redirect(
            url_for(
                "conversation_history.history_index",
                error="WhatsApp number is required.",
            )
        )

    if not message:
        return redirect(
            url_for(
                "conversation_history.history_index",
                wa_id=wa_id,
                error="Message content cannot be empty.",
            )
        )

    payload = get_text_message_input(wa_id, message)
    logging.info("Manual send triggered for %s", wa_id)
    _simulate_human_typing(wa_id)
    result = send_message(payload)

    status_code = None
    error_detail = None

    if isinstance(result, tuple):
        response_obj, status_code = result
        try:
            error_json = response_obj.get_json()
        except Exception:  # noqa: BLE001
            error_json = None
        if error_json and error_json.get("message"):
            error_detail = error_json["message"]
            if error_json.get("detail"):
                error_detail += f" ({error_json['detail']})"
    elif hasattr(result, "status_code"):
        status_code = result.status_code

    if status_code is not None and status_code >= 400:
        message_suffix = f"(status {status_code})"
        if error_detail:
            message_suffix += f": {error_detail}"
        return redirect(
            url_for(
                "conversation_history.history_index",
                wa_id=wa_id,
                error=f"Failed to send message {message_suffix}. Check server logs for details.",
            )
        )

    manual_entry = {
        "role": MANUAL_MESSAGE_ROLE,
        "content": message,
        "timestamp": _timestamp(),
        "ai_readable": False,
    }
    try:
        state = _load_thread_state(wa_id)
    except OSError as exc:
        logging.warning("Unable to load thread state for %s: %s", wa_id, exc)
        state = _coerce_thread_state(wa_id, [])
    state["messages"].append(manual_entry)

    try:
        _save_thread_state(wa_id, state)
    except OSError as exc:
        logging.warning("Failed to persist manual message for %s: %s", wa_id, exc)

    return redirect(
        url_for(
            "conversation_history.history_index",
            wa_id=wa_id,
            status=f"Message sent to {wa_id}.",
        )
    )


@history_blueprint.route("/toggle", methods=["POST"])
def toggle_ai():
    wa_id = normalize_wa_id(request.form.get("wa_id") or "")
    action = (request.form.get("action") or "").strip().lower()
    reason = (request.form.get("reason") or "").strip()

    if not wa_id:
        return redirect(
            url_for(
                "conversation_history.history_index",
                error="WhatsApp number is required to update automation state.",
            )
        )

    timestamp = _timestamp()

    with _open_thread_store() as db:
        state = _coerce_thread_state(wa_id, db.get(wa_id))
        if action == "resume":
            state["ai_paused"] = False
            state["handoff_reason"] = ""
            system_note = "Terima kasih sudah berbicara dengan tim Optimaxx. Sekarang balasan otomatis/Asisten Virtual Optimaxx telah diaktifkan kembali. 😊"
            _simulate_human_typing(wa_id)
            send_message(get_text_message_input(wa_id, system_note))
            upsert_contact(
                wa_id,
                allow_bot=True,
                source="dashboard_toggle",
            )
        else:
            state["ai_paused"] = True
            state["handoff_reason"] = reason or "Paused from dashboard"
            system_note = (
                "Balasan otomatis telah dijeda secara manual oleh tim Optimaxx."
            )
            _simulate_human_typing(wa_id)
            send_message(get_text_message_input(wa_id, system_note))
            upsert_contact(
                wa_id,
                allow_bot=False,
                source="dashboard_toggle",
            )
        state["handoff_ts"] = timestamp
        state["messages"].append(
            {
                "role": "system",
                "content": system_note,
                "timestamp": timestamp,
                "ai_readable": False,
            }
        )
        db[wa_id] = {
            "messages": state["messages"],
            "ai_paused": state["ai_paused"],
            "handoff_reason": state["handoff_reason"],
            "handoff_ts": state["handoff_ts"],
        }

    status_message = (
        f"Automation resumed for {wa_id}."
        if action == "resume"
        else f"Automation paused for {wa_id}. The model will not read new messages."
    )

    return redirect(
        url_for(
            "conversation_history.history_index",
            wa_id=wa_id,
            status=status_message,
        )
    )


@history_blueprint.route("/category", methods=["POST"])
def update_category():
    wa_id = normalize_wa_id(request.form.get("wa_id") or "")
    category = (request.form.get("category") or "").strip().lower()
    valid_categories = {"lead", "vendor", "other", "academic"}
    if not wa_id or category not in valid_categories:
        return redirect(
            url_for(
                "conversation_history.history_index",
                wa_id=wa_id,
                error="Invalid WhatsApp number or category.",
            )
        )

    allow_bot = category in {"lead", "academic"}
    updated = upsert_contact(
        wa_id,
        category=category,
        allow_bot=allow_bot,
        source="dashboard",
    )
    status_message = (
        f"Category for {wa_id} set to {category}."
        if updated
        else f"Unable to update category for {wa_id}."
    )
    return redirect(
        url_for(
            "conversation_history.history_index",
            wa_id=wa_id,
            status=status_message,
        )
    )


def create_app(url_prefix: str = ""):
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )
    # Ensure WAHA config/env values are loaded so manual sends work.
    load_configurations(app)
    configure_logging()
    app.register_blueprint(history_blueprint, url_prefix=url_prefix)
    return app
