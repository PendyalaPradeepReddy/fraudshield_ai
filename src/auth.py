"""
auth.py — Local user authentication for FraudShield AI.
Uses a JSON file database with SHA-256 salted password hashing.
No external auth service required.
"""

import json
import hashlib
import secrets
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.json")


# ── Database helpers ───────────────────────────────────────────────────────────

def _load_db() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    try:
        with open(DB_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_db(db: dict):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


# ── Password hashing ───────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Return (hashed_password, salt). Generate salt if not provided."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return hashed, salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    hashed, _ = _hash_password(password, salt)
    return hashed == stored_hash


# ── User management ────────────────────────────────────────────────────────────

def register_user(username: str, password: str, full_name: str = "", email: str = "") -> dict:
    """
    Create a new user account.
    Returns {"success": True} or {"success": False, "error": str}.
    """
    if not username or not password:
        return {"success": False, "error": "Username and password are required."}
    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}
    if len(username) < 3:
        return {"success": False, "error": "Username must be at least 3 characters."}

    db = _load_db()
    if username.lower() in {k.lower() for k in db}:
        return {"success": False, "error": "Username already exists. Please choose another."}

    hashed, salt = _hash_password(password)
    db[username] = {
        "username":   username,
        "full_name":  full_name,
        "email":      email,
        "password":   hashed,
        "salt":       salt,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "settings":   {},   # stores email_config, whatsapp_config etc.
    }
    _save_db(db)
    return {"success": True}


def login_user(username: str, password: str) -> dict:
    """
    Verify credentials.
    Returns {"success": True, "user": dict} or {"success": False, "error": str}.
    """
    if not username or not password:
        return {"success": False, "error": "Please enter username and password."}

    db = _load_db()
    # Case-insensitive username lookup
    match = next((v for k, v in db.items() if k.lower() == username.lower()), None)
    if not match:
        return {"success": False, "error": "Invalid username or password."}

    if not _verify_password(password, match["password"], match["salt"]):
        return {"success": False, "error": "Invalid username or password."}

    # Update last login
    match["last_login"] = datetime.now().isoformat()
    db[match["username"]] = match
    _save_db(db)

    return {"success": True, "user": {
        "username":  match["username"],
        "full_name": match.get("full_name", ""),
        "email":     match.get("email", ""),
        "created_at":match.get("created_at", ""),
        "last_login":match["last_login"],
    }}


def save_user_settings(username: str, settings: dict):
    """Save per-user settings (email_config, whatsapp_config, etc.)."""
    db = _load_db()
    if username in db:
        db[username]["settings"] = settings
        _save_db(db)


def load_user_settings(username: str) -> dict:
    """Load per-user settings."""
    db = _load_db()
    return db.get(username, {}).get("settings", {})


def change_password(username: str, old_password: str, new_password: str) -> dict:
    """Change user password after verifying old one."""
    db = _load_db()
    user = db.get(username)
    if not user:
        return {"success": False, "error": "User not found."}
    if not _verify_password(old_password, user["password"], user["salt"]):
        return {"success": False, "error": "Current password is incorrect."}
    if len(new_password) < 6:
        return {"success": False, "error": "New password must be at least 6 characters."}
    hashed, salt = _hash_password(new_password)
    user["password"] = hashed
    user["salt"]     = salt
    db[username]     = user
    _save_db(db)
    return {"success": True}


def get_all_users() -> list:
    """Return list of usernames (for admin use)."""
    db = _load_db()
    return [{"username": v["username"], "full_name": v.get("full_name",""),
             "created_at": v.get("created_at",""), "last_login": v.get("last_login","")}
            for v in db.values()]
