"""
WSGI entry point for production servers (gunicorn, uWSGI).

Gunicorn loads this module and serves the `app` object. We also warm the
pipeline cache here so the first real request doesn't eat the cold-start
hit (~3-5s).

Run locally:
    gunicorn wsgi:app

On Render: this is the `startCommand` in render.yaml.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make the project root importable regardless of CWD.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from webapp.app import app, _pipeline_state

# Warm the cache at import time so the first HTTP request is snappy.
# This runs once per gunicorn worker at boot.
print("[wsgi] warming pipeline cache...", flush=True)
try:
    _pipeline_state()
    print("[wsgi] ready", flush=True)
except Exception as exc:
    # Don't crash the worker if warmup fails — let requests see the error.
    print(f"[wsgi] warmup failed: {exc}", flush=True)
