"""
notes.py - Notes Management Routes

Handles note upload, viewing, and deletion operations.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from psycopg2.extras import RealDictCursor
from datetime import datetime
import os
import json
from database import get_db_connection
from utils import extract_text_from_pdf, extract_text_from_txt
from groq_service import add_note_to_index

# ===================== Blueprint ===================== #

notes_bp = Blueprint('notes', __name__)

# ===================== Helper ===================== #

def get_note_or_none(note_id, user_id):
    """Fetch a note belonging to user, return dict or None."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM notes WHERE id = %s AND user_id = %s',
                (note_id, user_id)
            )
            note = cur.fetchone()
    return dict(note) if note else None

# ===================== Routes ===================== #

@notes_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Upload a new note (PDF or TXT file)."""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        file = request.files.get('file')

        if not title or not file or file.filename == '':
            flash('Please provide title and select a file', 'danger')
            return redirect(url_for('notes.upload'))

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        if ext not in ['pdf', 'txt']:
            flash('Only PDF and TXT files are allowed', 'danger')
            return redirect(url_for('notes.upload'))

        from flask import current_app
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        content = extract_text_from_pdf(file_path) if ext == 'pdf' else extract_text_from_txt(file_path)
        upload_ts = datetime.utcnow()

        # FIX: use current_user.id
        user_id = int(current_user.id)

        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'INSERT INTO notes (user_id, title, content, file_type, upload_date) VALUES (%s, %s, %s, %s, %s) RETURNING id',
                        (user_id, title, content, ext, upload_ts)
                    )
                    note_id = cur.fetchone()['id']

            try:
                add_note_to_index(note_id, content)
            except Exception as e:
                print(f"RAG index error: {e}")

            flash('Note uploaded successfully!', 'success')
            return redirect(url_for('notes.notes'))
        except Exception as e:
            flash(f'Error saving note: {str(e)}', 'danger')
            return redirect(url_for('notes.upload'))

    return render_template('upload.html')


@notes_bp.route('/notes')
@login_required
def notes():
    """Display all notes for the current user."""
    user_id = int(current_user.id)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM notes WHERE user_id = %s ORDER BY upload_date DESC',
                (user_id,)
            )
            all_notes = cur.fetchall()

    notes_list = []
    for note in all_notes:
        note_copy = dict(note)
        ud = note_copy.get('upload_date')
        if isinstance(ud, str):
            try:
                note_copy['upload_date'] = datetime.fromisoformat(ud)
            except Exception:
                note_copy['upload_date'] = ud
        notes_list.append(note_copy)

    return render_template('notes.html', notes=notes_list)


@notes_bp.route('/note/<int:note_id>')
@login_required
def view_note(note_id):
    """View a specific note."""
    user_id = int(current_user.id)
    note_dict = get_note_or_none(note_id, user_id)

    if not note_dict:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))

    ud = note_dict.get('upload_date')
    if isinstance(ud, str):
        try:
            note_dict['upload_date'] = datetime.fromisoformat(ud)
        except Exception:
            pass

    if note_dict.get('questions'):
        try:
            note_dict['questions'] = json.loads(note_dict['questions'])
        except Exception:
            note_dict['questions'] = []
    else:
        note_dict['questions'] = []

    return render_template('view_note.html', note=note_dict)


@notes_bp.route('/delete/<int:note_id>', methods=['POST'])
@login_required
def delete_note(note_id):
    """Delete a note."""
    user_id = int(current_user.id)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'DELETE FROM notes WHERE id = %s AND user_id = %s',
                (note_id, user_id)
            )

    flash('Note deleted successfully!', 'success')
    return redirect(url_for('notes.notes'))


@notes_bp.route('/note/<int:note_id>/complete', methods=['POST'])
@login_required
def mark_completed(note_id):
    """Toggle note completion status."""
    user_id = int(current_user.id)
    note_dict = get_note_or_none(note_id, user_id)

    if not note_dict:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))

    new_status = not note_dict.get('is_completed', False)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE notes SET is_completed = %s WHERE id = %s AND user_id = %s',
                (new_status, note_id, user_id)
            )

    flash('Note status updated!', 'success')
    return redirect(url_for('notes.notes'))