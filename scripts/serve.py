"""Runs the inference API. python scripts/serve.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from src.api.app import create_app
from src.api.config import load_serve_config

ROOT = Path(__file__).resolve().parents[1]
config = load_serve_config(ROOT / "configs" / "serve.yaml")
app = create_app(config)

if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
