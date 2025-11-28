# ===================== app.py ===================== #
import json
from rag_service import answer_with_context
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    jsonify, make_response, send_file, session
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, UserMixin, current_user
)
import os
from datetime import datetime, date, timedelta
import io
import csv
from collections import OrderedDict

# PDF and Text Processing
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# LLM / AI Services
from llm_service import (
    summarize_text, generate_questions_from_text,
    explain_topic, add_note_to_index, rag_answer
)

# PostgreSQL
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import errors as pg_errors

# --------------------- Config --------------------- #
# Update this to your real DB connection string
DATABASE_URL = "postgresql://postgres:1107@localhost:5432/study_coach"

# ===================== Flask App Config ===================== #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===================== Flask-Login Setup ===================== #
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# ===================== Database Helper ===================== #
def get_db_connection():
    """
    Returns a psycopg2 connection with RealDictCursor.
    Use as: with get_db_connection() as conn:
             cur = conn.cursor(cursor_factory=RealDictCursor)
    """
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    """Initialize PostgreSQL database tables"""
    create_users = '''
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(150) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    );
    '''
    create_notes = '''
    CREATE TABLE IF NOT EXISTS notes (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        title VARCHAR(255) NOT NULL,
        content TEXT NOT NULL,
        file_type VARCHAR(50) NOT NULL,
        upload_date TIMESTAMP NOT NULL,
        summary TEXT,
        questions TEXT
    );
    '''
    create_exams = '''
    CREATE TABLE IF NOT EXISTS exams (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        exam_date DATE NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    '''
    create_subjects = '''
    CREATE TABLE IF NOT EXISTS subjects (
        id SERIAL PRIMARY KEY,
        exam_id INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
        subject_name VARCHAR(255) NOT NULL,
        chapters TEXT,
        priority VARCHAR(50)
    );
    '''
    create_schedules = '''
    CREATE TABLE IF NOT EXISTS schedules (
        id SERIAL PRIMARY KEY,
        exam_id INTEGER REFERENCES exams(id) ON DELETE CASCADE,
        date DATE NOT NULL,
        slot_start TIME NOT NULL,
        slot_end TIME NOT NULL,
        subject VARCHAR(255) NOT NULL,
        chapter TEXT,
        duration_minutes INTEGER NOT NULL,
        created_by VARCHAR(50) NOT NULL DEFAULT 'auto',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    '''
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_users)
            cur.execute(create_notes)
            cur.execute(create_exams)
            cur.execute(create_subjects)
            cur.execute(create_schedules)
        # conn commits on normal exit of context manager

init_db()

# ===================== Flask-Login User Class ===================== #
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = str(id)  # flask-login expects id to be str-like
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    """
    user_id will be a string; convert to int for DB lookup
    """
    try:
        uid = int(user_id)
    except Exception:
        return None

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM users WHERE id = %s', (uid,))
            row = cur.fetchone()

    if row:
        return User(row['id'], row['username'], row['password'])
    return None

# ===================== Helper Functions ===================== #
PRIORITY_WEIGHT = {'High': 1.5, 'Medium': 1.0, 'Low': 0.75}

def time_str_to_minutes(tstr):
    """Convert time string HH:MM to minutes."""
    h, m = map(int, tstr.split(':'))
    return h * 60 + m

def minutes_to_time_str(minutes):
    """Convert minutes to time string HH:MM."""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

def split_chapters(chapters_str):
    """Split comma-separated chapters string into list."""
    return [c.strip() for c in chapters_str.split(',') if c.strip()] if chapters_str else []

def assign_chapters_to_slots(subject_chapters_map, assignments):
    """Assign chapters to schedule slots sequentially."""
    chapters_lists = {s: list(chaps) for s, chaps in subject_chapters_map.items()}
    results = []
    for a in assignments:
        subj = a['subject']
        chapter = chapters_lists[subj].pop(0) if subj in chapters_lists and chapters_lists[subj] else ''
        row = a.copy()
        row['chapter'] = chapter
        results.append(row)
    return results

def extract_text_from_pdf(file_path):
    """Extract text content from PDF file."""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''.join((page.extract_text() or '') + '\n' for page in reader.pages)
            return text.strip() or "No text found in PDF."
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_txt(file_path):
    """Extract text content from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        return f"Error reading text file: {str(e)}"

# ===================== Routes: User Authentication ===================== #
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Username and password required', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id',
                        (username, hashed_password)
                    )
                    new = cur.fetchone()
                    # commit happens automatically on context exit
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            # Unique violation handling
            if isinstance(e, pg_errors.UniqueViolation) or getattr(e, 'pgcode', None) == pg_errors.UniqueViolation.__name__:
                flash('Username already exists', 'danger')
            else:
                flash(f'Error during registration: {str(e)}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT * FROM users WHERE username = %s', (username,))
                row = cur.fetchone()

        if row and check_password_hash(row['password'], password):
            user = User(row['id'], row['username'], row['password'])
            login_user(user)
            session['user_id'] = row['id']
            session['username'] = row['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

# ===================== Routes: Home ===================== #
@app.route('/')
@app.route('/home')
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

# ===================== Routes: Notes ===================== #
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        file = request.files.get('file')

        if not title or not file or file.filename == '':
            flash('Please provide title and select a file', 'danger')
            return redirect(url_for('upload'))

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if ext not in ['pdf', 'txt']:
            flash('Only PDF and TXT files are allowed', 'danger')
            return redirect(url_for('upload'))

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
            return redirect(url_for('notes'))
        except Exception as e:
            flash(f'Error saving note: {str(e)}', 'danger')
            return redirect(url_for('upload'))

    return render_template('upload.html')

@app.route('/notes')
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

@app.route('/note/<int:note_id>')
@login_required
def view_note(note_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM notes WHERE id = %s AND user_id = %s', (note_id, session['user_id']))
            note = cur.fetchone()

    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

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

@app.route('/delete/<int:note_id>', methods=['POST'])
@login_required
def delete_note(note_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM notes WHERE id = %s AND user_id = %s', (note_id, session['user_id']))
    flash('Note deleted successfully!', 'success')
    return redirect(url_for('notes'))

# ===================== Routes: AI Features ===================== #
@app.route('/summarize/<int:note_id>')
@login_required
def summarize_note(note_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM notes WHERE id = %s AND user_id = %s', (note_id, session['user_id']))
            note = cur.fetchone()
            if not note:
                flash('Note not found', 'danger')
                return redirect(url_for('notes'))

            note_dict = dict(note)
            summary = note_dict.get('summary')

            if not summary:
                try:
                    summary = summarize_text(note_dict['content'])
                    cur.execute('UPDATE notes SET summary = %s WHERE id = %s', (summary, note_id))
                except Exception as e:
                    flash(f'Error generating summary: {str(e)}', 'danger')
                    summary = None

    return render_template('summary.html', note=note_dict, summary=summary)

@app.route('/questions/<int:note_id>', methods=['GET', 'POST'])
@login_required
def questions_route(note_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM notes WHERE id = %s AND user_id = %s', (note_id, session['user_id']))
            note = cur.fetchone()

            if not note:
                flash('Note not found', 'danger')
                return redirect(url_for('notes'))

            note_dict = dict(note)
            questions_data = note_dict.get('questions')

            try:
                questions = json.loads(questions_data) if questions_data else []
            except Exception:
                questions = []

            if not questions:
                try:
                    questions = generate_questions_from_text(note_dict['content'])
                    cur.execute('UPDATE notes SET questions = %s WHERE id = %s', (json.dumps(questions), note_id))
                except Exception as e:
                    flash(f'Error generating questions: {str(e)}', 'danger')
                    questions = []

    print("Generated Questions:", questions)
    return render_template('questions.html', note=note_dict, questions=questions)

@app.route('/explain/<int:note_id>', methods=['GET', 'POST'])
@login_required
def explain_note(note_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM notes WHERE id = %s AND user_id = %s', (note_id, session['user_id']))
            note = cur.fetchone()

    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

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

@app.route('/ask_note/<int:note_id>', methods=['POST'])
@login_required
def ask_note(note_id):
    query = request.form.get('query')
    if not query:
        flash('Please enter a question.', 'warning')
        return redirect(url_for('view_note', note_id=note_id))

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM notes WHERE id = %s AND user_id = %s', (note_id, session['user_id']))
            note = cur.fetchone()

    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

    note_text = note['content']
    answer = answer_with_context(note_text, query)

    return render_template(
        'view_note.html',
        note=dict(note),
        query=query,
        answer=answer
    )

# ===================== Routes: RAG Chat ===================== #
@app.route('/rag_chat')
@login_required
def rag_chat():
    return render_template('rag_chat.html')

@app.route('/ask_rag', methods=['POST'])
@login_required
def ask_rag():
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'answer': 'Please enter a question.'})
    
    try:
        answer = rag_answer(query)
    except Exception as e:
        answer = f"Error during RAG: {str(e)}"
    
    return jsonify({'answer': answer})

# ===================== Routes: Scheduling ===================== #
@app.route('/create_schedule', methods=['GET', 'POST'])
@login_required
def create_schedule():
    if request.method == 'POST':
        exam_name = request.form.get('exam_name', '').strip()
        exam_date = request.form.get('exam_date', '').strip()
        start_time = request.form.get('start_time', '09:00').strip()
        hours_per_day = int(request.form.get('hours_per_day', 4))
        
        subjects_names = request.form.getlist('subject_name[]')
        subjects_chapters = request.form.getlist('chapters[]')
        subjects_priority = request.form.getlist('priority[]')

        if not exam_name or not exam_date:
            flash('Please provide exam name and date', 'danger')
            return redirect(url_for('create_schedule'))

        try:
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'INSERT INTO exams (name, exam_date, created_at) VALUES (%s, %s, %s) RETURNING id',
                        (exam_name, exam_date, datetime.utcnow())
                    )
                    exam_id = cur.fetchone()['id']

                    # Save subjects
                    for name, ch, pr in zip(subjects_names, subjects_chapters, subjects_priority):
                        if name.strip():
                            cur.execute(
                                'INSERT INTO subjects (exam_id, subject_name, chapters, priority) VALUES (%s, %s, %s, %s)',
                                (exam_id, name.strip(), ch.strip(), pr)
                            )

                    # fetch subjects back
                    cur.execute('SELECT * FROM subjects WHERE exam_id = %s', (exam_id,))
                    subjects_rows = cur.fetchall()

            # Build mapping for schedule generation (done outside connection to avoid nested DB calls)
            subj_chapters_map = {s['subject_name']: split_chapters(s['chapters']) for s in subjects_rows}
            priorities = {s['subject_name']: s.get('priority') or 'Medium' for s in subjects_rows}
            subject_list = list(subj_chapters_map.keys())

            if not subject_list:
                flash('Please add at least one subject', 'danger')
                return redirect(url_for('create_schedule'))

            today_date = date.today()
            exam_day = datetime.fromisoformat(exam_date).date()
            total_days = (exam_day - today_date).days
            
            if total_days <= 0:
                flash('Exam date must be in the future', 'danger')
                return redirect(url_for('create_schedule'))

            weight_map = {subj: PRIORITY_WEIGHT.get(priorities[subj], 1.0) for subj in subject_list}
            schedule_slots = []

            for d in range(total_days):
                day_date = today_date + timedelta(days=d)
                ordered_subjects = sorted(subject_list, key=lambda s: -weight_map.get(s, 1.0))
                total_weight = sum(weight_map[s] for s in ordered_subjects)
                total_minutes = hours_per_day * 60
                
                subj_minutes = {
                    s: int(round((weight_map[s] / total_weight) * total_minutes)) 
                    for s in ordered_subjects
                }
                
                diff = total_minutes - sum(subj_minutes.values())
                if ordered_subjects:
                    subj_minutes[ordered_subjects[0]] += diff

                cur_min = time_str_to_minutes(start_time)
                for subj in ordered_subjects:
                    minutes_for_subj = subj_minutes[subj]
                    while minutes_for_subj > 0:
                        slot_len = min(60, minutes_for_subj)
                        schedule_slots.append({
                            'exam_id': exam_id,
                            'date': day_date.strftime('%Y-%m-%d'),
                            'slot_start': minutes_to_time_str(cur_min),
                            'slot_end': minutes_to_time_str(cur_min + slot_len),
                            'subject': subj,
                            'duration_minutes': slot_len,
                            'created_by': 'auto'
                        })
                        cur_min += slot_len
                        minutes_for_subj -= slot_len

            assigned_with_chapters = assign_chapters_to_slots(subj_chapters_map, schedule_slots)

            # Save into schedules table
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    for row in assigned_with_chapters:
                        cur.execute('''
                            INSERT INTO schedules
                            (exam_id, date, slot_start, slot_end, subject, chapter, duration_minutes, created_by, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            row['exam_id'], row['date'], row['slot_start'], row['slot_end'],
                            row['subject'], row.get('chapter', ''), row['duration_minutes'],
                            'auto', datetime.utcnow()
                        ))
            flash('Auto schedule generated and saved!', 'success')
            return redirect(url_for('view_schedule', exam_id=exam_id))
        except Exception as e:
            flash(f'Error creating schedule: {str(e)}', 'danger')
            return redirect(url_for('create_schedule'))

    return render_template('create_schedule.html')

@app.route('/schedules')
@login_required
def all_schedules():
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM exams ORDER BY exam_date')
            exams = cur.fetchall()
    return render_template('all_schedules.html', exams=exams)

@app.route('/schedule/<int:exam_id>')
@login_required
def view_schedule(exam_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM exams WHERE id = %s', (exam_id,))
            exam = cur.fetchone()
            cur.execute('SELECT * FROM schedules WHERE exam_id = %s ORDER BY date, slot_start', (exam_id,))
            rows = cur.fetchall()

    if not exam:
        flash('Exam not found', 'danger')
        return redirect(url_for('all_schedules'))

    grouped = OrderedDict()
    for r in rows:
        grouped.setdefault(r['date'].isoformat() if isinstance(r['date'], (datetime,)) else r['date'], []).append(dict(r))

    return render_template('view_schedule.html', exam=exam, schedule=grouped, exam_id=exam_id)

@app.route('/schedule/<int:exam_id>/delete', methods=['POST'])
@login_required
def delete_schedule(exam_id):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM schedules WHERE exam_id = %s', (exam_id,))
            cur.execute('DELETE FROM subjects WHERE exam_id = %s', (exam_id,))
            cur.execute('DELETE FROM exams WHERE id = %s', (exam_id,))
    flash('Schedule and exam deleted successfully!', 'success')
    return redirect(url_for('all_schedules'))

@app.route('/schedule/<int:exam_id>/export/csv')
@login_required
def export_schedule_csv(exam_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT date, slot_start, slot_end, subject, chapter, duration_minutes '
                'FROM schedules WHERE exam_id = %s ORDER BY date, slot_start', 
                (exam_id,)
            )
            rows = cur.fetchall()

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Start', 'End', 'Subject', 'Chapter', 'Duration (min)'])
    for r in rows:
        date_val = r['date'].isoformat() if isinstance(r['date'], (datetime,)) else r['date']
        cw.writerow([date_val, r['slot_start'], r['slot_end'], r['subject'], r.get('chapter', ''), r['duration_minutes']])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=exam_{exam_id}_schedule.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/schedule/<int:exam_id>/export/pdf')
@login_required
def export_schedule_pdf(exam_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM exams WHERE id = %s', (exam_id,))
            exam = cur.fetchone()
            cur.execute('SELECT * FROM schedules WHERE exam_id = %s ORDER BY date, slot_start', (exam_id,))
            rows = cur.fetchall()

    if not exam or not rows:
        flash('No schedule found to export', 'danger')
        return redirect(url_for('view_schedule', exam_id=exam_id))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title=f"Study Schedule - {exam['name']}")
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph(f"Study Schedule: {exam['name']}", styles['Title']))
    elements.append(Paragraph(f"Exam Date: {exam['exam_date']}", styles['Normal']))
    elements.append(Spacer(1, 12))

    grouped = OrderedDict()
    for r in rows:
        date_key = r['date'].isoformat() if isinstance(r['date'], (datetime,)) else r['date']
        grouped.setdefault(date_key, []).append(r)

    for date_str, day_rows in grouped.items():
        elements.append(Paragraph(f"<b>Date: {date_str}</b>", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        data = [['Start', 'End', 'Subject', 'Chapter', 'Duration (min)']]
        for r in day_rows:
            data.append([r['slot_start'], r['slot_end'], r['subject'], r.get('chapter', '') or '', r['duration_minutes']])
        
        table = Table(data, colWidths=[60, 60, 120, 140, 80], repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d6efd')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, 
                    download_name=f"exam_{exam_id}_schedule.pdf", 
                    mimetype='application/pdf')

# ===================== Context Processor ===================== #
@app.context_processor
def inject_now():
    return {'datetime': datetime, 'current_user': current_user}

# ===================== Run App ===================== #
if __name__ == '__main__':
    app.run(debug=True)
