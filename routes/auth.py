from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from pymongo.errors import DuplicateKeyError
from database import get_db
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
            db = get_db()
            db.users.insert_one({'username': username, 'password': hashed_password})
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('auth.login'))
        except DuplicateKeyError:
            flash('Username already exists', 'danger')
            return redirect(url_for('auth.register'))
        except Exception as e:
            flash(f'Error during registration: {str(e)}', 'danger')
            return redirect(url_for('auth.register'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        db = get_db()
        row = db.users.find_one({'username': username})

        if row and check_password_hash(row['password'], password):
            user = User(str(row['_id']), row['username'], row['password'])
            login_user(user)
            session['user_id'] = str(row['_id'])
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