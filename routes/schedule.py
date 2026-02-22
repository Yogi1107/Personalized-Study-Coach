"""
schedule.py - Study Schedule Routes
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, make_response, send_file
from flask_login import login_required, current_user
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

schedule_bp = Blueprint('schedule', __name__)


def get_exam_or_none(exam_id, user_id):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM exams WHERE id = %s AND user_id = %s', (exam_id, user_id))
            exam = cur.fetchone()
    return dict(exam) if exam else None


@schedule_bp.route('/create_schedule', methods=['GET', 'POST'])
@login_required
def create_schedule():
    if request.method == 'POST':
        user_id = int(current_user.id)
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
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        'INSERT INTO exams (user_id, name, exam_date, created_at) VALUES (%s, %s, %s, %s) RETURNING id',
                        (user_id, exam_name, exam_date, datetime.utcnow())
                    )
                    exam_id = cur.fetchone()['id']

                    for name, ch, pr in zip(subjects_names, subjects_chapters, subjects_priority):
                        if name.strip():
                            cur.execute(
                                'INSERT INTO subjects (exam_id, subject_name, chapters, priority) VALUES (%s, %s, %s, %s)',
                                (exam_id, name.strip(), ch.strip(), pr)
                            )

                    cur.execute('SELECT * FROM subjects WHERE exam_id = %s', (exam_id,))
                    subjects_rows = cur.fetchall()

            subj_chapters_map = {s['subject_name']: split_chapters(s['chapters']) for s in subjects_rows}
            priorities = {s['subject_name']: s.get('priority') or 'Medium' for s in subjects_rows}
            subject_list = list(subj_chapters_map.keys())

            if not subject_list:
                flash('Please add at least one subject', 'danger')
                return redirect(url_for('schedule.create_schedule'))

            today_date = date.today()
            exam_day = datetime.fromisoformat(exam_date).date()
            total_days = (exam_day - today_date).days

            if total_days <= 0:
                flash('Exam date must be in the future', 'danger')
                return redirect(url_for('schedule.create_schedule'))

            weight_map = {subj: PRIORITY_WEIGHT.get(priorities[subj], 1.0) for subj in subject_list}
            total_weight = sum(weight_map.values())
            total_minutes_per_day = hours_per_day * 60

            # FIX: compute per-subject minutes once, rounded to clean 15-min blocks
            # This avoids fractional leftovers that produce 9-min junk slots
            BLOCK = 15  # minimum slot granularity in minutes
            raw_minutes = {
                s: (weight_map[s] / total_weight) * total_minutes_per_day
                for s in subject_list
            }
            # Round each subject to nearest BLOCK
            subj_minutes = {
                s: max(BLOCK, round(raw_minutes[s] / BLOCK) * BLOCK)
                for s in subject_list
            }
            # Fix rounding so total equals total_minutes_per_day exactly
            allocated = sum(subj_minutes.values())
            diff = total_minutes_per_day - allocated
            # Apply leftover diff to highest-priority subject
            top_subj = max(subject_list, key=lambda s: weight_map[s])
            subj_minutes[top_subj] = max(BLOCK, subj_minutes[top_subj] + diff)

            schedule_slots = []
            ordered_subjects = sorted(subject_list, key=lambda s: -weight_map[s])

            for d in range(total_days):
                day_date = today_date + timedelta(days=d)
                cur_min = time_str_to_minutes(start_time)

                for subj in ordered_subjects:
                    minutes_remaining = subj_minutes[subj]
                    while minutes_remaining > 0:
                        # FIX: use 60-min slots, last slot gets remainder (≥ BLOCK)
                        slot_len = min(60, minutes_remaining)
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
                        minutes_remaining -= slot_len

            # Assign chapters (cycling) and merge tiny slots
            assigned_with_chapters = assign_chapters_to_slots(subj_chapters_map, schedule_slots)

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

            flash('Schedule generated and saved!', 'success')
            return redirect(url_for('schedule.view_schedule', exam_id=exam_id))

        except Exception as e:
            flash(f'Error creating schedule: {str(e)}', 'danger')
            return redirect(url_for('schedule.create_schedule'))

    return render_template('create_schedule.html')


@schedule_bp.route('/schedules')
@login_required
def all_schedules():
    user_id = int(current_user.id)
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT * FROM exams WHERE user_id = %s ORDER BY exam_date', (user_id,))
            exams = cur.fetchall()
    return render_template('all_schedules.html', exams=exams)


@schedule_bp.route('/schedule/<int:exam_id>')
@login_required
def view_schedule(exam_id):
    user_id = int(current_user.id)
    exam = get_exam_or_none(exam_id, user_id)
    if not exam:
        flash('Schedule not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM schedules WHERE exam_id = %s ORDER BY date, slot_start',
                (exam_id,)
            )
            rows = cur.fetchall()

    grouped = OrderedDict()
    for r in rows:
        date_key = r['date'].isoformat() if hasattr(r['date'], 'isoformat') else str(r['date'])
        grouped.setdefault(date_key, []).append(dict(r))

    return render_template('view_schedule.html', exam=exam, schedule=grouped, exam_id=exam_id)


@schedule_bp.route('/schedule/<int:exam_id>/delete', methods=['POST'])
@login_required
def delete_schedule(exam_id):
    user_id = int(current_user.id)
    if not get_exam_or_none(exam_id, user_id):
        flash('Schedule not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM schedules WHERE exam_id = %s', (exam_id,))
            cur.execute('DELETE FROM subjects WHERE exam_id = %s', (exam_id,))
            cur.execute('DELETE FROM exams WHERE id = %s', (exam_id,))

    flash('Schedule deleted successfully!', 'success')
    return redirect(url_for('schedule.all_schedules'))


@schedule_bp.route('/schedule/<int:exam_id>/export/csv')
@login_required
def export_schedule_csv(exam_id):
    user_id = int(current_user.id)
    if not get_exam_or_none(exam_id, user_id):
        flash('Schedule not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))

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
        date_val = r['date'].isoformat() if hasattr(r['date'], 'isoformat') else str(r['date'])
        cw.writerow([date_val, r['slot_start'], r['slot_end'], r['subject'], r.get('chapter', ''), r['duration_minutes']])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=exam_{exam_id}_schedule.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@schedule_bp.route('/schedule/<int:exam_id>/export/pdf')
@login_required
def export_schedule_pdf(exam_id):
    user_id = int(current_user.id)
    exam = get_exam_or_none(exam_id, user_id)
    if not exam:
        flash('Schedule not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM schedules WHERE exam_id = %s ORDER BY date, slot_start',
                (exam_id,)
            )
            rows = cur.fetchall()

    if not rows:
        flash('No schedule found to export', 'danger')
        return redirect(url_for('schedule.view_schedule', exam_id=exam_id))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title=f"Study Schedule - {exam['name']}")
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"Study Schedule: {exam['name']}", styles['Title']))
    elements.append(Paragraph(f"Exam Date: {exam['exam_date']}", styles['Normal']))
    elements.append(Spacer(1, 12))

    grouped = OrderedDict()
    for r in rows:
        date_key = r['date'].isoformat() if hasattr(r['date'], 'isoformat') else str(r['date'])
        grouped.setdefault(date_key, []).append(r)

    for date_str, day_rows in grouped.items():
        elements.append(Paragraph(f"Date: {date_str}", styles['Heading2']))
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
    return send_file(buffer, as_attachment=True, download_name=f"exam_{exam_id}_schedule.pdf", mimetype='application/pdf')