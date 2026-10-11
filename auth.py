"""
auth.py
=======
Authentication & Session Management Module
Enforces strict separation between regular user sessions and administrative sessions:
- Regular users: Authenticated against local SQLite database (users.db) with bcrypt/scrypt hashed passwords.
- Admin: Authenticated via hardcoded admin username and environment-variable-backed ADMIN_PASSWORD.
- Independent session keys: session['user_id'] vs session['admin_id'].
- Custom decorators: @login_required (user routes) and @admin_required (ops routes).
"""

import os
import sqlite3
from functools import wraps
from flask import session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

# Admin credentials
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "adminpass123")


def get_db_connection():
    """Establishes a connection to the SQLite users database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the users table and seeds initial demo users if not present."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Seed default demo regular user if database is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        seed_users = [
            ("user@gmail.com", "user123"),
            ("trader@gmail.com", "trader123")
        ]
        for email, pwd in seed_users:
            p_hash = generate_password_hash(pwd)
            cursor.execute("INSERT OR IGNORE INTO users (email, password_hash) VALUES (?, ?)", (email, p_hash))
        conn.commit()
    conn.close()


def register_user(email, password):
    """Registers a new regular user with hashed password."""
    email = email.strip().lower()
    if not email or "@" not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        p_hash = generate_password_hash(password)
        cursor.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, p_hash))
        conn.commit()
        return True, "Registration successful! You may now sign in."
    except sqlite3.IntegrityError:
        return False, "An account with that email already exists."
    finally:
        conn.close()


def authenticate_user(email, password):
    """Validates user credentials against stored hashes."""
    email = email.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        return {"id": user["id"], "email": user["email"]}
    return None


def authenticate_admin(username, password):
    """Validates administrative credentials strictly from environment/configuration."""
    username = username.strip()
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return {"username": ADMIN_USERNAME}
    return None


def login_required(f):
    """Decorator ensuring that an active user session exists."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access the trader dashboard.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator ensuring that an active administrator session exists."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_id"):
            flash("Administrator authentication required to access this area.", "danger")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function
