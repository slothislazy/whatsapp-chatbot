import sys
import os
from dotenv import load_dotenv
import logging


def load_configurations(app):
    load_dotenv()

    # WAHA (WhatsApp HTTP API) settings
    app.config["WAHA_BASE_URL"] = (os.getenv("WAHA_BASE_URL") or "http://localhost:3000/").rstrip(
        "/"
    )
    app.config["WAHA_API_KEY"] = os.getenv("WAHA_API_KEY")
    app.config["WAHA_SESSION"] = os.getenv("WAHA_SESSION") or "default"

    workers_raw = os.getenv("WHATSAPP_WORKERS", "").strip()
    try:
        workers = int(workers_raw)
    except ValueError:
        workers = 1
    app.config["WHATSAPP_WORKERS"] = max(1, workers)

    max_age_raw = os.getenv("WHATSAPP_MAX_MESSAGE_AGE_SECONDS", "").strip()
    try:
        max_age = int(max_age_raw)
    except ValueError:
        max_age = 0
    app.config["WHATSAPP_MAX_MESSAGE_AGE_SECONDS"] = max(0, max_age)


def configure_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
