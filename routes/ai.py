"""
ai.py - AI Features Routes
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from bson import ObjectId
import json
from database import get_db
from groq_service import summarize_text, generate_questions_from_text, explain_topic
from rag_service import answer_with_context, rag_answer

ai_bp = Blueprint('ai', __name__)

# ===================== Helper ===================== #

def get_note_or_none(note_id, user_id):
    try:
        oid = ObjectId(note_id)
    except Exception:
        return None
    db = get_db()
    note = db.notes.find_one({'_id': oid, 'user_id': user_id})
    return note  # already a dict; None if not found

# ===================== Single-Note AI Features ===================== #

@ai_bp.route('/summarize/<note_id>')
@login_required
def summarize_note(note_id):
    user_id = current_user.id  # already a string
    note_dict = get_note_or_none(note_id, user_id)

    if not note_dict:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))

    summary = note_dict.get('summary')

    if not summary:
        try:
            summary = summarize_text(note_dict['content'])
            db = get_db()
            db.notes.update_one(
                {'_id': note_dict['_id']},
                {'$set': {'summary': summary}}
            )
        except Exception as e:
            flash(f'Error generating summary: {str(e)}', 'danger')
            summary = None

    return render_template('summary.html', note=note_dict, summary=summary)


@ai_bp.route('/questions/<note_id>', methods=['GET', 'POST'])
@login_required
def questions_route(note_id):
    user_id = current_user.id
    note_dict = get_note_or_none(note_id, user_id)

    if not note_dict:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))

    questions_data = note_dict.get('questions')

    try:
        questions = json.loads(questions_data) if questions_data else []
    except Exception:
        questions = []

    if request.method == 'POST':
        completed_indices = set(request.form.getlist('completed[]'))
        for i, q in enumerate(questions):
            q['completed'] = str(i) in completed_indices
        db = get_db()
        db.notes.update_one(
            {'_id': note_dict['_id']},
            {'$set': {'questions': json.dumps(questions)}}
        )
        flash('Progress saved!', 'success')
        return redirect(url_for('ai.questions_route', note_id=note_id))

    if not questions:
        try:
            questions = generate_questions_from_text(note_dict['content'])
            db = get_db()
            db.notes.update_one(
                {'_id': note_dict['_id']},
                {'$set': {'questions': json.dumps(questions)}}
            )
        except Exception as e:
            flash(f'Error generating questions: {str(e)}', 'danger')
            questions = []

    return render_template('questions.html', note=note_dict, questions=questions)


@ai_bp.route('/explain/<note_id>', methods=['GET', 'POST'])
@login_required
def explain_note(note_id):
    user_id = current_user.id
    note_dict = get_note_or_none(note_id, user_id)

    if not note_dict:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))

    explanation = None

    if request.method == 'POST':
        user_question = request.form.get('question', '').strip()
        if user_question:
            try:
                explanation = explain_topic(note_dict['content'], user_question)
            except Exception as e:
                flash(f'Error generating explanation: {str(e)}', 'danger')

    return render_template('explain.html', note=note_dict, explanation=explanation)


@ai_bp.route('/ask_note/<note_id>', methods=['POST'])
@login_required
def ask_note(note_id):
    query = request.form.get('query')
    if not query:
        flash('Please enter a question.', 'warning')
        return redirect(url_for('notes.view_note', note_id=note_id))

    user_id = current_user.id
    note_dict = get_note_or_none(note_id, user_id)

    if not note_dict:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))

    answer = answer_with_context(note_dict['content'], query)

    return render_template(
        'view_note.html',
        note=note_dict,
        query=query,
        answer=answer
    )

# ===================== Multi-Note RAG Chat ===================== #

@ai_bp.route('/rag_chat')
@login_required
def rag_chat():
    return render_template('rag_chat.html')


@ai_bp.route('/ask_rag', methods=['POST'])
@login_required
def ask_rag():
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'answer': 'Please enter a question.'})

    try:
        answer = rag_answer(query, user_id=current_user.id)
    except Exception as e:
        answer = f"Error during RAG: {str(e)}"

    return jsonify({'answer': answer})