from flask import Blueprint, render_template, session
from flask_login import login_required, current_user
from psycopg2.extras import RealDictCursor
from database import get_db_connection

# ===================== Routes: Home ===================== #

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@main_bp.route('/home')
@login_required
def home():
    # FIX: use current_user.id instead of session
    user_id = int(current_user.id)

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT COUNT(*) as cnt FROM notes WHERE user_id = %s', (user_id,))
            total_notes = cur.fetchone()['cnt']

            cur.execute('SELECT COUNT(*) as cnt FROM notes WHERE user_id = %s AND summary IS NOT NULL', (user_id,))
            total_summaries = cur.fetchone()['cnt']

            cur.execute('SELECT COUNT(*) as cnt FROM notes WHERE user_id = %s AND questions IS NOT NULL', (user_id,))
            total_questions = cur.fetchone()['cnt']

            # FIX: count only this user's schedules via exams.user_id
            cur.execute('SELECT COUNT(*) as cnt FROM exams WHERE user_id = %s', (user_id,))
            total_schedules = cur.fetchone()['cnt']

    return render_template(
        'home.html',
        username=current_user.username,
        total_notes=total_notes,
        total_summaries=total_summaries,
        total_questions=total_questions,
        total_schedules=total_schedules
    )