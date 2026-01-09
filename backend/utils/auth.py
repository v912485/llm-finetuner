import os
from functools import wraps
from flask import request, jsonify


def _get_admin_token() -> str | None:
    token = os.environ.get("ADMIN_TOKEN")
    return token.strip() if token else None


def is_request_authenticated() -> bool:
    admin_token = _get_admin_token()
    if not admin_token:
        return False
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {admin_token}"


def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        admin_token = _get_admin_token()
        if not admin_token:
            return (
                jsonify({"status": "error", "message": "Server not configured: ADMIN_TOKEN missing"}),
                500,
            )

        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {admin_token}":
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        return fn(*args, **kwargs)

    return wrapper







