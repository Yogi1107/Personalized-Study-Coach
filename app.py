from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
import PyPDF2
from llm_service import summarize_text, generate_questions_from_text, explain_topic
from dotenv import load_dotenv

# -------------------- Load Environment -------------------- #
load_dotenv()
hf_token = os.getenv("HF_API_TOKEN")
print("HF Token loaded:", hf_token[:4] + "****")  # for testing

# -------------------- Flask App Config -------------------- #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DATABASE = 'study_coach.db'

# -------------------- Database -------------------- #
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                file_type TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                summary TEXT,
                questions TEXT
            )
        ''')
        conn.commit()

init_db()

# -------------------- Helper Functions -------------------- #
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''.join(page.extract_text() + '\n' for page in reader.pages if page.extract_text())
            return text.strip() or "No text found in PDF."
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        return f"Error reading text file: {str(e)}"

# -------------------- Routes -------------------- #
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        title = request.form.get('title')
        file = request.files.get('file')

        if not title:
            flash('Please provide a note title', 'danger')
            return redirect(url_for('upload'))
        if not file or file.filename == '':
            flash('Please select a file', 'danger')
            return redirect(url_for('upload'))

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext not in ['pdf', 'txt']:
            flash('Only PDF and TXT files are allowed', 'danger')
            return redirect(url_for('upload'))

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        content = extract_text_from_pdf(file_path) if ext == 'pdf' else extract_text_from_txt(file_path)

        conn = get_db_connection()
        conn.execute(
            'INSERT INTO notes (title, content, file_type, upload_date) VALUES (?, ?, ?, ?)',
            (title, content, ext, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

        flash('Note uploaded successfully!', 'success')
        return redirect(url_for('notes'))

    return render_template('upload.html')

@app.route('/notes')
def notes():
    conn = get_db_connection()
    all_notes = conn.execute('SELECT * FROM notes ORDER BY upload_date DESC').fetchall()
    conn.close()
    notes_list = [dict(note, upload_date=datetime.fromisoformat(note['upload_date'])) for note in all_notes]
    return render_template('notes.html', notes=notes_list)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    conn = get_db_connection()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

    note_dict = dict(note)
    note_dict['upload_date'] = datetime.fromisoformat(note['upload_date'])
    return render_template('view_note.html', note=note_dict)

@app.route('/delete/<int:note_id>', methods=['POST'])
def delete_note(note_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    flash('Note deleted successfully!', 'success')
    return redirect(url_for('notes'))

# -------------------- AI Routes -------------------- #

# Summarize note
@app.route('/summarize/<int:note_id>')
def summarize_note(note_id):
    conn = get_db_connection()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    if not note:
        conn.close()
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

    note_dict = dict(note)

    try:
        # Use chunked summarization
        summary = summarize_text(note_dict['content'])
        conn.execute('UPDATE notes SET summary = ? WHERE id = ?', (summary, note_id))
        conn.commit()
    except Exception as e:
        flash(f'Error generating summary: {str(e)}', 'danger')
        summary = None

    conn.close()
    return render_template('summary.html', note=note_dict, summary=summary)


# Generate questions route
@app.route('/questions/<int:note_id>')
def questions_route(note_id):
    conn = get_db_connection()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    if not note:
        conn.close()
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

    note_dict = dict(note)

    try:
        # Chunked question generation
        questions = generate_questions_from_text(note_dict['content'])
        conn.execute('UPDATE notes SET questions = ? WHERE id = ?', (questions, note_id))
        conn.commit()
    except Exception as e:
        flash(f'Error generating questions: {str(e)}', 'danger')
        questions = None

    conn.close()
    return render_template('questions.html', note=note_dict, questions=questions)


# Explain topic route
@app.route('/explain/<int:note_id>', methods=['GET', 'POST'])
def explain_note(note_id):
    conn = get_db_connection()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

    note_dict = dict(note)
    explanation = None

    if request.method == 'POST':
        user_question = request.form.get('question')
        if user_question:
            try:
                # Use chunked explanation
                explanation = explain_topic(note_dict['content'], user_question)
            except Exception as e:
                flash(f'Error generating explanation: {str(e)}', 'danger')

    return render_template('explain.html', note=note_dict, explanation=explanation)


# -------------------- Inject datetime -------------------- #
@app.context_processor
def inject_now():
    return {'datetime': datetime}

# -------------------- Run App -------------------- #
if __name__ == '__main__':
    app.run(debug=True)
