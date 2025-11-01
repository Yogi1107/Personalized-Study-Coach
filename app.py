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
import sqlite3
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

# ===================== Flask App Config ===================== #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
DATABASE = 'study_coach.db'

# ===================== Flask-Login Setup ===================== #
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


# ===================== Database Helper ===================== #
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    with get_db_connection() as conn:
        # Users table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')

        # Notes table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                file_type TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                summary TEXT,
                questions TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Exams table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                exam_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')

        # Subjects table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                subject_name TEXT NOT NULL,
                chapters TEXT,
                priority TEXT,
                FOREIGN KEY (exam_id) REFERENCES exams(id)
            )
        ''')

        # Schedules table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER,
                date TEXT NOT NULL,
                slot_start TEXT NOT NULL,
                slot_end TEXT NOT NULL,
                subject TEXT NOT NULL,
                chapter TEXT,
                duration_minutes INTEGER NOT NULL,
                created_by TEXT NOT NULL DEFAULT 'auto',
                created_at TEXT NOT NULL,
                FOREIGN KEY (exam_id) REFERENCES exams(id)
            )
        ''')
        conn.commit()


init_db()


# ===================== Flask-Login User Class ===================== #
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
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
            text = ''.join(page.extract_text() + '\n' for page in reader.pages if page.extract_text())
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
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                        (username, hashed_password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists', 'danger')
        finally:
            conn.close()
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = get_db_connection()
        row = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

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
    conn = get_db_connection()

    total_notes = conn.execute(
        'SELECT COUNT(*) FROM notes WHERE user_id = ?', (user_id,)
    ).fetchone()[0]
    
    total_summaries = conn.execute(
        'SELECT COUNT(*) FROM notes WHERE user_id = ? AND summary IS NOT NULL', (user_id,)
    ).fetchone()[0]
    
    total_questions = conn.execute(
        'SELECT COUNT(*) FROM notes WHERE user_id = ? AND questions IS NOT NULL', (user_id,)
    ).fetchone()[0]
    
    total_schedules = conn.execute(
        'SELECT COUNT(*) FROM schedules'
    ).fetchone()[0]

    conn.close()

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

        conn = get_db_connection()
        cur = conn.execute(
            'INSERT INTO notes (user_id, title, content, file_type, upload_date) VALUES (?, ?, ?, ?, ?)',
            (session['user_id'], title, content, ext, datetime.utcnow().isoformat())
        )
        note_id = cur.lastrowid
        conn.commit()
        conn.close()

        try:
            add_note_to_index(note_id, content)
        except Exception as e:
            print(f"RAG index error: {e}")

        flash('Note uploaded successfully!', 'success')
        return redirect(url_for('notes'))

    return render_template('upload.html')


@app.route('/notes')
@login_required
def notes():
    conn = get_db_connection()
    all_notes = conn.execute(
        'SELECT * FROM notes WHERE user_id = ? ORDER BY upload_date DESC', 
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    notes_list = [
        dict(note, upload_date=datetime.fromisoformat(note['upload_date'])) 
        for note in all_notes
    ]
    return render_template('notes.html', notes=notes_list)


@app.route('/note/<int:note_id>')
@login_required
def view_note(note_id):
    conn = get_db_connection()
    note = conn.execute(
        'SELECT * FROM notes WHERE id = ? AND user_id = ?', 
        (note_id, session['user_id'])
    ).fetchone()
    conn.close()
    
    if not note:
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))
    
    note_dict = dict(note)
    note_dict['upload_date'] = datetime.fromisoformat(note['upload_date'])

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
    conn = get_db_connection()
    conn.execute('DELETE FROM notes WHERE id = ? AND user_id = ?', 
                (note_id, session['user_id']))
    conn.commit()
    conn.close()
    flash('Note deleted successfully!', 'success')
    return redirect(url_for('notes'))


# ===================== Routes: AI Features ===================== #
@app.route('/summarize/<int:note_id>')
@login_required
def summarize_note(note_id):
    conn = get_db_connection()
    note = conn.execute(
        'SELECT * FROM notes WHERE id = ? AND user_id = ?', 
        (note_id, session['user_id'])
    ).fetchone()
    
    if not note:
        conn.close()
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

    note_dict = dict(note)
    summary = note_dict.get('summary')
    
    if not summary:
        try:
            summary = summarize_text(note_dict['content'])
            conn.execute('UPDATE notes SET summary = ? WHERE id = ?', (summary, note_id))
            conn.commit()
        except Exception as e:
            flash(f'Error generating summary: {str(e)}', 'danger')
            summary = None
    
    conn.close()
    return render_template('summary.html', note=note_dict, summary=summary)


@app.route('/questions/<int:note_id>', methods=['GET', 'POST'])
@login_required
def questions_route(note_id):
    conn = get_db_connection()
    note = conn.execute(
        'SELECT * FROM notes WHERE id = ? AND user_id = ?',
        (note_id, session['user_id'])
    ).fetchone()

    if not note:
        conn.close()
        flash('Note not found', 'danger')
        return redirect(url_for('notes'))

    note_dict = dict(note)
    questions_data = note_dict.get('questions')

    # Try to load stored questions
    try:
        questions = json.loads(questions_data) if questions_data else []
    except Exception:
        questions = []

    # Generate if not found
    if not questions:
        try:
            questions = generate_questions_from_text(note_dict['content'])
            conn.execute(
                'UPDATE notes SET questions = ? WHERE id = ?',
                (json.dumps(questions), note_id)
            )
            conn.commit()
        except Exception as e:
            flash(f'Error generating questions: {str(e)}', 'danger')
            questions = []

    conn.close()

    print("Generated Questions:", questions)
    return render_template('questions.html', note=note_dict, questions=questions)


@app.route('/explain/<int:note_id>', methods=['GET', 'POST'])
@login_required
def explain_note(note_id):
    conn = get_db_connection()
    note = conn.execute(
        'SELECT * FROM notes WHERE id = ? AND user_id = ?', 
        (note_id, session['user_id'])
    ).fetchone()
    conn.close()
    
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

    conn = get_db_connection()
    note = conn.execute(
        'SELECT * FROM notes WHERE id = ? AND user_id = ?',
        (note_id, session['user_id'])
    ).fetchone()
    conn.close()

    if not note:
        flash('Note not found.', 'danger')
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

        conn = get_db_connection()
        cur = conn.execute(
            'INSERT INTO exams (name, exam_date, created_at) VALUES (?, ?, ?)',
            (exam_name, exam_date, datetime.utcnow().isoformat())
        )
        exam_id = cur.lastrowid

        # Save subjects
        for name, ch, pr in zip(subjects_names, subjects_chapters, subjects_priority):
            if name.strip():
                conn.execute(
                    'INSERT INTO subjects (exam_id, subject_name, chapters, priority) VALUES (?, ?, ?, ?)',
                    (exam_id, name.strip(), ch.strip(), pr)
                )
        conn.commit()

        # Build mapping for schedule generation
        subjects_rows = conn.execute(
            'SELECT * FROM subjects WHERE exam_id = ?', (exam_id,)
        ).fetchall()
        
        subj_chapters_map = {s['subject_name']: split_chapters(s['chapters']) for s in subjects_rows}
        priorities = {s['subject_name']: s['priority'] or 'Medium' for s in subjects_rows}
        subject_list = list(subj_chapters_map.keys())

        if not subject_list:
            conn.close()
            flash('Please add at least one subject', 'danger')
            return redirect(url_for('create_schedule'))

        # Generate schedule slots
        today_date = date.today()
        exam_day = datetime.fromisoformat(exam_date).date()
        total_days = (exam_day - today_date).days
        
        if total_days <= 0:
            conn.close()
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

        # Assign chapters sequentially
        assigned_with_chapters = assign_chapters_to_slots(subj_chapters_map, schedule_slots)

        # Save into schedules table
        for row in assigned_with_chapters:
            conn.execute('''
                INSERT INTO schedules 
                (exam_id, date, slot_start, slot_end, subject, chapter, duration_minutes, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (row['exam_id'], row['date'], row['slot_start'], row['slot_end'], 
                 row['subject'], row.get('chapter', ''), row['duration_minutes'], 
                 'auto', datetime.utcnow().isoformat())
            )
        conn.commit()
        conn.close()
        
        flash('Auto schedule generated and saved!', 'success')
        return redirect(url_for('view_schedule', exam_id=exam_id))

    return render_template('create_schedule.html')


@app.route('/schedules')
@login_required
def all_schedules():
    conn = get_db_connection()
    exams = conn.execute('SELECT * FROM exams ORDER BY exam_date').fetchall()
    conn.close()
    return render_template('all_schedules.html', exams=exams)


@app.route('/schedule/<int:exam_id>')
@login_required
def view_schedule(exam_id):
    conn = get_db_connection()
    exam = conn.execute('SELECT * FROM exams WHERE id = ?', (exam_id,)).fetchone()
    rows = conn.execute(
        'SELECT * FROM schedules WHERE exam_id = ? ORDER BY date, slot_start', 
        (exam_id,)
    ).fetchall()
    conn.close()

    if not exam:
        flash('Exam not found', 'danger')
        return redirect(url_for('all_schedules'))

    grouped = OrderedDict()
    for r in rows:
        grouped.setdefault(r['date'], []).append(dict(r))

    return render_template('view_schedule.html', exam=exam, schedule=grouped, exam_id=exam_id)


@app.route('/schedule/<int:exam_id>/delete', methods=['POST'])
@login_required
def delete_schedule(exam_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM schedules WHERE exam_id = ?', (exam_id,))
    conn.execute('DELETE FROM subjects WHERE exam_id = ?', (exam_id,))
    conn.execute('DELETE FROM exams WHERE id = ?', (exam_id,))
    conn.commit()
    conn.close()
    flash('Schedule and exam deleted successfully!', 'success')
    return redirect(url_for('all_schedules'))


@app.route('/schedule/<int:exam_id>/export/csv')
@login_required
def export_schedule_csv(exam_id):
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT date, slot_start, slot_end, subject, chapter, duration_minutes '
        'FROM schedules WHERE exam_id = ? ORDER BY date, slot_start', 
        (exam_id,)
    ).fetchall()
    conn.close()

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Start', 'End', 'Subject', 'Chapter', 'Duration (min)'])
    for r in rows:
        cw.writerow([r['date'], r['slot_start'], r['slot_end'], 
                    r['subject'], r['chapter'] or '', r['duration_minutes']])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=exam_{exam_id}_schedule.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@app.route('/schedule/<int:exam_id>/export/pdf')
@login_required
def export_schedule_pdf(exam_id):
    conn = get_db_connection()
    exam = conn.execute('SELECT * FROM exams WHERE id = ?', (exam_id,)).fetchone()
    rows = conn.execute(
        'SELECT * FROM schedules WHERE exam_id = ? ORDER BY date, slot_start', 
        (exam_id,)
    ).fetchall()
    conn.close()

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
        grouped.setdefault(r['date'], []).append(r)

    for date_str, day_rows in grouped.items():
        elements.append(Paragraph(f"<b>Date: {date_str}</b>", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        data = [['Start', 'End', 'Subject', 'Chapter', 'Duration (min)']]
        for r in day_rows:
            data.append([r['slot_start'], r['slot_end'], r['subject'], 
                        r['chapter'] or '', r['duration_minutes']])
        
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