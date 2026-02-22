from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from psycopg2.extras import RealDictCursor
from psycopg2 import errors as pg_errors
from database import get_db_connection
from models import User

# ===================== Routes: User Authentication ===================== #

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Username and password required', 'danger')
            return redirect(url_for('auth.register'))

        hashed_password = generate_password_hash(password)

        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id',
                        (username, hashed_password)
                    )
                    cur.fetchone()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            # FIX: simplified unique violation check
            if isinstance(e, pg_errors.UniqueViolation):
                flash('Username already exists', 'danger')
            else:
                flash(f'Error during registration: {str(e)}', 'danger')
            return redirect(url_for('auth.register'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT * FROM users WHERE username = %s', (username,))
                row = cur.fetchone()

        if row and check_password_hash(row['password'], password):
            user = User(row['id'], row['username'], row['password'])
            login_user(user)
            # FIX: use current_user going forward; keep session for templates that need it
            session['user_id'] = row['id']
            session['username'] = row['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('main.home'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('auth.login'))