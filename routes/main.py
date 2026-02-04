from flask import Blueprint, render_template, session
from flask_login import login_required
from psycopg2.extras import RealDictCursor
from database import get_db_connection

# ===================== Routes: Home ===================== #

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@main_bp.route('/home')
@login_required
def home():
    user_id = session.get('user_id')
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT COUNT(*) as cnt FROM notes WHERE user_id = %s', (user_id,))
            total_notes = cur.fetchone()['cnt']
            
            cur.execute('SELECT COUNT(*) as cnt FROM notes WHERE user_id = %s AND summary IS NOT NULL', (user_id,))
            total_summaries = cur.fetchone()['cnt']
            
            cur.execute('SELECT COUNT(*) as cnt FROM notes WHERE user_id = %s AND questions IS NOT NULL', (user_id,))
            total_questions = cur.fetchone()['cnt']
            
            cur.execute('SELECT COUNT(*) as cnt FROM schedules')
            total_schedules = cur.fetchone()['cnt']
    
    return render_template(
        'home.html',
        username=session.get('username', 'User'),
        total_notes=total_notes,
        total_summaries=total_summaries,
        total_questions=total_questions,
        total_schedules=total_schedules
    )