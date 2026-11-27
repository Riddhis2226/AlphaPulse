"""Production WSGI entrypoint for AlphaPulse (Gunicorn / Render / Docker)."""

from app import server as application

__all__ = ["application"]