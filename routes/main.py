from flask import Blueprint, render_template, session
from flask_login import login_required, current_user
from database import get_db

# ===================== Routes: Home ===================== #

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
@main_bp.route('/home')
@login_required
def home():
    user_id = current_user.id  # already a string, no int() cast needed

    db = get_db()

    total_notes     = db.notes.count_documents({'user_id': user_id})
    total_summaries = db.notes.count_documents({'user_id': user_id, 'summary': {'$ne': None}})
    total_questions = db.notes.count_documents({'user_id': user_id, 'questions': {'$ne': None}})
    total_schedules = db.exams.count_documents({'user_id': user_id})

    return render_template(
        'home.html',
        username=current_user.username,
        total_notes=total_notes,
        total_summaries=total_summaries,
        total_questions=total_questions,
        total_schedules=total_schedules
    )