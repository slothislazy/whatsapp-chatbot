import logging

from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest

from .utils.whatsapp_utils import (
    process_whatsapp_message,
    is_valid_whatsapp_message,
)
from app.services.whatsapp_queue import enqueue_whatsapp_job

webhook_blueprint = Blueprint("webhook", __name__)


def handle_message():
    """
    Handle incoming WAHA webhook events (message/message.any) and enqueue for processing.
    """
    try:
        body = request.get_json()
    except BadRequest:
        logging.error("Failed to parse JSON payload", exc_info=True)
        return jsonify({"status": "error", "message": "Invalid JSON provided"}), 400

    if not isinstance(body, dict):
        logging.error("Invalid JSON payload: expected JSON object")
        return jsonify({"status": "error", "message": "Invalid JSON provided"}), 400

    try:
        if is_valid_whatsapp_message(body):
            _enqueue_whatsapp_processing(body)
            return jsonify({"status": "ok"}), 200
        else:
            return (
                jsonify({"status": "error", "message": "Not a WhatsApp API event"}),
                404,
            )
    except (KeyError, TypeError):
        logging.error("Unexpected payload structure from WhatsApp", exc_info=True)
        return jsonify({"status": "error", "message": "Invalid payload structure"}), 400


def _enqueue_whatsapp_processing(body):
    try:
        enqueue_whatsapp_job(body)
    except Exception:
        logging.exception("Failed to enqueue WhatsApp payload; processing immediately.")
        process_whatsapp_message(body)


@webhook_blueprint.route("/webhook", methods=["POST"])
def webhook_post():
    return handle_message()
