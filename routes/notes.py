"""
notes.py - Notes Management Routes

Handles note upload, viewing, and deletion operations.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from bson import ObjectId
from datetime import datetime
import os
import json
from database import get_db
from utils import extract_text_from_pdf, extract_text_from_txt
from groq_service import add_note_to_index

# ===================== Blueprint ===================== #

notes_bp = Blueprint('notes', __name__)

# ===================== Helper ===================== #

def get_note_or_none(note_id, user_id):
    """Fetch a note belonging to user, return dict or None."""
    try:
        oid = ObjectId(note_id)
    except Exception:
        return None
    db = get_db()
    return db.notes.find_one({'_id': oid, 'user_id': user_id})

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

        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        content = extract_text_from_pdf(file_path) if ext == 'pdf' else extract_text_from_txt(file_path)

        try:
            db = get_db()
            result = db.notes.insert_one({
                'user_id': current_user.id,
                'title': title,
                'content': content,
                'file_type': ext,
                'upload_date': datetime.utcnow(),
                'summary': None,
                'questions': None,
                'is_completed': False
            })

            try:
                add_note_to_index(str(result.inserted_id), content)
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
    db = get_db()
    all_notes = list(db.notes.find(
        {'user_id': current_user.id},
        sort=[('upload_date', -1)]
    ))
    return render_template('notes.html', notes=all_notes)


@notes_bp.route('/note/<note_id>')
@login_required
def view_note(note_id):
    """View a specific note."""
    note_dict = get_note_or_none(note_id, current_user.id)

    if not note_dict:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))

    if note_dict.get('questions'):
        try:
            note_dict['questions'] = json.loads(note_dict['questions'])
        except Exception:
            note_dict['questions'] = []
    else:
        note_dict['questions'] = []

    return render_template('view_note.html', note=note_dict)


@notes_bp.route('/delete/<note_id>', methods=['POST'])
@login_required
def delete_note(note_id):
    """Delete a note."""
    try:
        oid = ObjectId(note_id)
    except Exception:
        flash('Invalid note ID', 'danger')
        return redirect(url_for('notes.notes'))

    db = get_db()
    db.notes.delete_one({'_id': oid, 'user_id': current_user.id})

    flash('Note deleted successfully!', 'success')
    return redirect(url_for('notes.notes'))


@notes_bp.route('/note/<note_id>/complete', methods=['POST'])
@login_required
def mark_completed(note_id):
    """Toggle note completion status."""
    note_dict = get_note_or_none(note_id, current_user.id)

    if not note_dict:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))

    new_status = not note_dict.get('is_completed', False)
    db = get_db()
    db.notes.update_one(
        {'_id': note_dict['_id']},
        {'$set': {'is_completed': new_status}}
    )

    flash('Note status updated!', 'success')
    return redirect(url_for('notes.notes'))