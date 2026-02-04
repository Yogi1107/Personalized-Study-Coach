from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_required
from werkzeug.utils import secure_filename
from psycopg2.extras import RealDictCursor
from datetime import datetime
import os
import json
from database import get_db_connection
from utils import extract_text_from_pdf, extract_text_from_txt
from llm_service import add_note_to_index

# ===================== Routes: Notes ===================== #

notes_bp = Blueprint('notes', __name__)


@notes_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
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
        
        # Get upload folder from app config
        from flask import current_app
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        content = extract_text_from_pdf(file_path) if ext == 'pdf' else extract_text_from_txt(file_path)
        upload_ts = datetime.utcnow()
        
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'INSERT INTO notes (user_id, title, content, file_type, upload_date) VALUES (%s, %s, %s, %s, %s) RETURNING id',
                        (session['user_id'], title, content, ext, upload_ts)
                    )
                    new = cur.fetchone()
                    note_id = new['id']
            
            # add to RAG index asynchronously if possible; here we call and ignore errors
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
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM notes WHERE user_id = %s ORDER BY upload_date DESC', (session['user_id'],))
            all_notes = cur.fetchall()
    
    notes_list = []
    for note in all_notes:
        note_copy = dict(note)
        # upload_date from Postgres is likely a datetime object already
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
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM notes WHERE id = %s AND user_id = %s', (note_id, session['user_id']))
            note = cur.fetchone()
    
    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes.notes'))
    
    note_dict = dict(note)
    ud = note_dict.get('upload_date')
    if isinstance(ud, str):
        try:
            note_dict['upload_date'] = datetime.fromisoformat(ud)
        except Exception:
            pass
    
    # Decode stored JSON questions safely
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
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM notes WHERE id = %s AND user_id = %s', (note_id, session['user_id']))
    
    flash('Note deleted successfully!', 'success')
    return redirect(url_for('notes.notes'))