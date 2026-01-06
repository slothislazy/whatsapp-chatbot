import logging
from queue import Queue
from threading import Thread
from typing import Any, Dict

_job_queue: "Queue[Dict[str, Any]]" = Queue()
_workers_started = False


def start_whatsapp_workers(app, num_workers: int = 1) -> None:
    """Start background workers that process WhatsApp jobs from the shared queue."""
    global _workers_started
    if _workers_started:
        return

    worker_count = max(1, int(num_workers or 1))
    for idx in range(worker_count):
        thread = Thread(
            target=_worker_loop,
            args=(app,),
            daemon=True,
            name=f"whatsapp-worker-{idx + 1}",
        )
        thread.start()
    _workers_started = True
    logging.info("Started %s WhatsApp worker(s).", worker_count)


def enqueue_whatsapp_job(payload: Dict[str, Any]) -> None:
    """Push a WhatsApp webhook payload onto the queue."""
    _job_queue.put(payload)


def _worker_loop(app) -> None:
    """Worker loop that drains the queue and processes each job."""
    from app.utils.whatsapp_utils import process_whatsapp_message

    while True:
        payload = _job_queue.get()
        try:
            with app.app_context():
                process_whatsapp_message(payload)
        except Exception:
            logging.exception("Failed to process WhatsApp message from queue")
        finally:
            _job_queue.task_done()
