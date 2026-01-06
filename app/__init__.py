from pathlib import Path

from flask import Flask

from app.config import configure_logging, load_configurations
from app.services.whatsapp_queue import start_whatsapp_workers

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


def create_app(url_prefix: str = ""):
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )

    load_configurations(app)
    configure_logging()

    from .views import webhook_blueprint
    from app.services.see_history import history_blueprint

    app.register_blueprint(webhook_blueprint)
    app.register_blueprint(history_blueprint, url_prefix=url_prefix)

    worker_count = app.config.get("WHATSAPP_WORKERS", 1)
    start_whatsapp_workers(app, worker_count)

    return app
