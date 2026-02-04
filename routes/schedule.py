"""
schedule.py - Study Schedule Routes

Handles exam and study schedule creation, viewing, and export.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, make_response, send_file
from flask_login import login_required
from psycopg2.extras import RealDictCursor
from datetime import datetime, date, timedelta
import io
import csv
from collections import OrderedDict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from database import get_db_connection
from utils import (
    PRIORITY_WEIGHT,
    time_str_to_minutes,
    minutes_to_time_str,
    split_chapters,
    assign_chapters_to_slots
)

# ===================== Blueprint ===================== #

schedule_bp = Blueprint('schedule', __name__)

# ===================== Routes ===================== #

@schedule_bp.route('/create_schedule', methods=['GET', 'POST'])
@login_required
def create_schedule():
    """Create a new study schedule for an exam."""
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
            return redirect(url_for('schedule.create_schedule'))
        
        try:
            # Save exam
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
                    
                    # Fetch subjects back
                    cur.execute('SELECT * FROM subjects WHERE exam_id = %s', (exam_id,))
                    subjects_rows = cur.fetchall()
            
            # Build mapping for schedule generation
            subj_chapters_map = {s['subject_name']: split_chapters(s['chapters']) for s in subjects_rows}
            priorities = {s['subject_name']: s.get('priority') or 'Medium' for s in subjects_rows}
            subject_list = list(subj_chapters_map.keys())
            
            if not subject_list:
                flash('Please add at least one subject', 'danger')
                return redirect(url_for('schedule.create_schedule'))
            
            # Calculate schedule
            today_date = date.today()
            exam_day = datetime.fromisoformat(exam_date).date()
            total_days = (exam_day - today_date).days
            
            if total_days <= 0:
                flash('Exam date must be in the future', 'danger')
                return redirect(url_for('schedule.create_schedule'))
            
            weight_map = {subj: PRIORITY_WEIGHT.get(priorities[subj], 1.0) for subj in subject_list}
            schedule_slots = []
            
            # Generate schedule for each day
            for d in range(total_days):
                day_date = today_date + timedelta(days=d)
                ordered_subjects = sorted(subject_list, key=lambda s: -weight_map.get(s, 1.0))
                total_weight = sum(weight_map[s] for s in ordered_subjects)
                total_minutes = hours_per_day * 60
                
                # Allocate time per subject based on priority
                subj_minutes = {
                    s: int(round((weight_map[s] / total_weight) * total_minutes))
                    for s in ordered_subjects
                }
                
                # Adjust for rounding errors
                diff = total_minutes - sum(subj_minutes.values())
                if ordered_subjects:
                    subj_minutes[ordered_subjects[0]] += diff
                
                # Create time slots
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
            
            # Assign chapters to slots
            assigned_with_chapters = assign_chapters_to_slots(subj_chapters_map, schedule_slots)
            
            # Save schedule to database
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    for row in assigned_with_chapters:
                        cur.execute('''
                            INSERT INTO schedules (exam_id, date, slot_start, slot_end, subject, chapter, duration_minutes, created_by, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                            row['exam_id'],
                            row['date'],
                            row['slot_start'],
                            row['slot_end'],
                            row['subject'],
                            row.get('chapter', ''),
                            row['duration_minutes'],
                            'auto',
                            datetime.utcnow()
                        ))
            
            flash('Auto schedule generated and saved!', 'success')
            return redirect(url_for('schedule.view_schedule', exam_id=exam_id))
        
        except Exception as e:
            flash(f'Error creating schedule: {str(e)}', 'danger')
            return redirect(url_for('schedule.create_schedule'))
    
    return render_template('create_schedule.html')


@schedule_bp.route('/schedules')
@login_required
def all_schedules():
    """Display all exam schedules."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM exams ORDER BY exam_date')
            exams = cur.fetchall()
    
    return render_template('all_schedules.html', exams=exams)


@schedule_bp.route('/schedule/<int:exam_id>')
@login_required
def view_schedule(exam_id):
    """View a specific exam schedule."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM exams WHERE id = %s', (exam_id,))
            exam = cur.fetchone()
            cur.execute(
                'SELECT * FROM schedules WHERE exam_id = %s ORDER BY date, slot_start',
                (exam_id,)
            )
            rows = cur.fetchall()
    
    if not exam:
        flash('Exam not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))
    
    # Group schedule by date
    grouped = OrderedDict()
    for r in rows:
        date_key = r['date'].isoformat() if isinstance(r['date'], datetime) else r['date']
        grouped.setdefault(date_key, []).append(dict(r))
    
    return render_template('view_schedule.html', exam=exam, schedule=grouped, exam_id=exam_id)


@schedule_bp.route('/schedule/<int:exam_id>/delete', methods=['POST'])
@login_required
def delete_schedule(exam_id):
    """Delete an exam schedule."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM schedules WHERE exam_id = %s', (exam_id,))
            cur.execute('DELETE FROM subjects WHERE exam_id = %s', (exam_id,))
            cur.execute('DELETE FROM exams WHERE id = %s', (exam_id,))
    
    flash('Schedule and exam deleted successfully!', 'success')
    return redirect(url_for('schedule.all_schedules'))


@schedule_bp.route('/schedule/<int:exam_id>/export/csv')
@login_required
def export_schedule_csv(exam_id):
    """Export schedule as CSV file."""
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
        date_val = r['date'].isoformat() if isinstance(r['date'], datetime) else r['date']
        cw.writerow([
            date_val,
            r['slot_start'],
            r['slot_end'],
            r['subject'],
            r.get('chapter', ''),
            r['duration_minutes']
        ])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=exam_{exam_id}_schedule.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@schedule_bp.route('/schedule/<int:exam_id>/export/pdf')
@login_required
def export_schedule_pdf(exam_id):
    """Export schedule as PDF file."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM exams WHERE id = %s', (exam_id,))
            exam = cur.fetchone()
            cur.execute(
                'SELECT * FROM schedules WHERE exam_id = %s ORDER BY date, slot_start',
                (exam_id,)
            )
            rows = cur.fetchall()
    
    if not exam or not rows:
        flash('No schedule found to export', 'danger')
        return redirect(url_for('schedule.view_schedule', exam_id=exam_id))
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title=f"Study Schedule - {exam['name']}")
    elements = []
    styles = getSampleStyleSheet()
    
    # Add title and exam info
    elements.append(Paragraph(f"Study Schedule: {exam['name']}", styles['Title']))
    elements.append(Paragraph(f"Exam Date: {exam['exam_date']}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Group schedule by date
    grouped = OrderedDict()
    for r in rows:
        date_key = r['date'].isoformat() if isinstance(r['date'], datetime) else r['date']
        grouped.setdefault(date_key, []).append(r)
    
    # Create table for each day
    for date_str, day_rows in grouped.items():
        elements.append(Paragraph(f"Date: {date_str}", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        data = [['Start', 'End', 'Subject', 'Chapter', 'Duration (min)']]
        for r in day_rows:
            data.append([
                r['slot_start'],
                r['slot_end'],
                r['subject'],
                r.get('chapter', '') or '',
                r['duration_minutes']
            ])
        
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
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"exam_{exam_id}_schedule.pdf",
        mimetype='application/pdf'
    )