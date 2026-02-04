"""
ai.py - AI Features Routes

Handles AI-powered features like summarization, question generation,
explanations, and RAG-based chat.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import login_required
from psycopg2.extras import RealDictCursor
import json
from database import get_db_connection
from llm_service import summarize_text, generate_questions_from_text, explain_topic
from rag_service import answer_with_context, rag_answer

# ===================== Blueprint ===================== #

ai_bp = Blueprint('ai', __name__)

# ===================== Single-Note AI Features ===================== #

@ai_bp.route('/summarize/<int:note_id>')
@login_required
def summarize_note(note_id):
    """Generate or display summary for a specific note."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM notes WHERE id = %s AND user_id = %s',
                (note_id, session['user_id'])
            )
            note = cur.fetchone()
    
    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))
    
    note_dict = dict(note)
    summary = note_dict.get('summary')
    
    # Generate summary if not already exists
    if not summary:
        try:
            summary = summarize_text(note_dict['content'])
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'UPDATE notes SET summary = %s WHERE id = %s',
                        (summary, note_id)
                    )
        except Exception as e:
            flash(f'Error generating summary: {str(e)}', 'danger')
            summary = None
    
    return render_template('summary.html', note=note_dict, summary=summary)


@ai_bp.route('/questions/<int:note_id>', methods=['GET', 'POST'])
@login_required
def questions_route(note_id):
    """Generate or display practice questions for a specific note."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM notes WHERE id = %s AND user_id = %s',
                (note_id, session['user_id'])
            )
            note = cur.fetchone()
    
    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))
    
    note_dict = dict(note)
    questions_data = note_dict.get('questions')
    
    # Parse existing questions
    try:
        questions = json.loads(questions_data) if questions_data else []
    except Exception:
        questions = []
    
    # Generate questions if not already exists
    if not questions:
        try:
            questions = generate_questions_from_text(note_dict['content'])
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        'UPDATE notes SET questions = %s WHERE id = %s',
                        (json.dumps(questions), note_id)
                    )
        except Exception as e:
            flash(f'Error generating questions: {str(e)}', 'danger')
            questions = []
    
    return render_template('questions.html', note=note_dict, questions=questions)


@ai_bp.route('/explain/<int:note_id>', methods=['GET', 'POST'])
@login_required
def explain_note(note_id):
    """Explain a specific topic or concept from a note."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM notes WHERE id = %s AND user_id = %s',
                (note_id, session['user_id'])
            )
            note = cur.fetchone()
    
    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))
    
    note_dict = dict(note)
    explanation = None
    
    if request.method == 'POST':
        user_question = request.form.get('question', '').strip()
        if user_question:
            try:
                explanation = explain_topic(note_dict['content'], user_question)
            except Exception as e:
                flash(f'Error generating explanation: {str(e)}', 'danger')
    
    return render_template('explain.html', note=note_dict, explanation=explanation)


@ai_bp.route('/ask_note/<int:note_id>', methods=['POST'])
@login_required
def ask_note(note_id):
    """Ask a question about a specific note using RAG."""
    query = request.form.get('query')
    if not query:
        flash('Please enter a question.', 'warning')
        return redirect(url_for('notes.view_note', note_id=note_id))
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM notes WHERE id = %s AND user_id = %s',
                (note_id, session['user_id'])
            )
            note = cur.fetchone()
    
    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))
    
    note_text = note['content']
    answer = answer_with_context(note_text, query)
    
    return render_template(
        'view_note.html',
        note=dict(note),
        query=query,
        answer=answer
    )

# ===================== Multi-Note RAG Chat ===================== #

@ai_bp.route('/rag_chat')
@login_required
def rag_chat():
    """RAG chat interface for querying across all notes."""
    return render_template('rag_chat.html')


@ai_bp.route('/ask_rag', methods=['POST'])
@login_required
def ask_rag():
    """Answer questions using RAG across all user notes."""
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'answer': 'Please enter a question.'})
    
    try:
        answer = rag_answer(query, user_id=session.get('user_id'))
    except Exception as e:
        answer = f"Error during RAG: {str(e)}"
    
    return jsonify({'answer': answer})