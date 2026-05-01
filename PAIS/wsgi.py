from __future__ import annotations
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from webapp.app import app, _pipeline_state


print("[wsgi] warming pipeline cache...", flush=True)
try:
    _pipeline_state()
    print("[wsgi] ready", flush=True)
except Exception as exc:

    print(f"[wsgi] warmup failed: {exc}", flush=True)
