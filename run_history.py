import logging
import os

from app.services.see_history import create_app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("HISTORY_HOST", "0.0.0.0")
    port = int(os.getenv("HISTORY_PORT", "5000"))
    create_app().run(host=host, port=port)
