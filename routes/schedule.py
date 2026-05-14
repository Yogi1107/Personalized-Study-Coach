"""
schedule.py - Study Schedule Routes
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, make_response, send_file
from flask_login import login_required, current_user
from bson import ObjectId
from datetime import datetime, date, timedelta
import io
import csv
from collections import OrderedDict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from database import get_db
from utils import (
    PRIORITY_WEIGHT,
    time_str_to_minutes,
    minutes_to_time_str,
    split_chapters,
    assign_chapters_to_slots
)

schedule_bp = Blueprint('schedule', __name__)


def get_exam_or_none(exam_id, user_id):
    try:
        oid = ObjectId(exam_id)
    except Exception:
        return None
    db = get_db()
    return db.exams.find_one({'_id': oid, 'user_id': user_id})


@schedule_bp.route('/create_schedule', methods=['GET', 'POST'])
@login_required
def create_schedule():
    if request.method == 'POST':
        user_id = current_user.id
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
            db = get_db()

            # Insert exam
            exam_result = db.exams.insert_one({
                'user_id': user_id,
                'name': exam_name,
                'exam_date': exam_date,
                'created_at': datetime.utcnow()
            })
            exam_id = exam_result.inserted_id

            # Insert subjects
            subjects_to_insert = [
                {
                    'exam_id': exam_id,
                    'subject_name': name.strip(),
                    'chapters': ch.strip(),
                    'priority': pr
                }
                for name, ch, pr in zip(subjects_names, subjects_chapters, subjects_priority)
                if name.strip()
            ]
            if subjects_to_insert:
                db.subjects.insert_many(subjects_to_insert)

            subjects_rows = list(db.subjects.find({'exam_id': exam_id}))

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

            BLOCK = 15
            raw_minutes = {
                s: (weight_map[s] / total_weight) * total_minutes_per_day
                for s in subject_list
            }
            subj_minutes = {
                s: max(BLOCK, round(raw_minutes[s] / BLOCK) * BLOCK)
                for s in subject_list
            }
            allocated = sum(subj_minutes.values())
            diff = total_minutes_per_day - allocated
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

            assigned_with_chapters = assign_chapters_to_slots(subj_chapters_map, schedule_slots)

            # Add created_at to each slot and bulk insert
            for row in assigned_with_chapters:
                row['created_at'] = datetime.utcnow()
            db.schedules.insert_many(assigned_with_chapters)

            flash('Schedule generated and saved!', 'success')
            return redirect(url_for('schedule.view_schedule', exam_id=str(exam_id)))

        except Exception as e:
            flash(f'Error creating schedule: {str(e)}', 'danger')
            return redirect(url_for('schedule.create_schedule'))

    return render_template('create_schedule.html')


@schedule_bp.route('/schedules')
@login_required
def all_schedules():
    db = get_db()
    exams = list(db.exams.find(
        {'user_id': current_user.id},
        sort=[('exam_date', 1)]
    ))
    return render_template('all_schedules.html', exams=exams)


@schedule_bp.route('/schedule/<exam_id>')
@login_required
def view_schedule(exam_id):
    exam = get_exam_or_none(exam_id, current_user.id)
    if not exam:
        flash('Schedule not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))

    db = get_db()
    rows = list(db.schedules.find(
        {'exam_id': exam['_id']},
        sort=[('date', 1), ('slot_start', 1)]
    ))

    grouped = OrderedDict()
    for r in rows:
        date_key = r['date'] if isinstance(r['date'], str) else r['date'].isoformat()
        grouped.setdefault(date_key, []).append(r)

    return render_template('view_schedule.html', exam=exam, schedule=grouped, exam_id=exam_id)


@schedule_bp.route('/schedule/<exam_id>/delete', methods=['POST'])
@login_required
def delete_schedule(exam_id):
    exam = get_exam_or_none(exam_id, current_user.id)
    if not exam:
        flash('Schedule not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))

    db = get_db()
    oid = exam['_id']
    db.schedules.delete_many({'exam_id': oid})
    db.subjects.delete_many({'exam_id': oid})
    db.exams.delete_one({'_id': oid})

    flash('Schedule deleted successfully!', 'success')
    return redirect(url_for('schedule.all_schedules'))


@schedule_bp.route('/schedule/<exam_id>/export/csv')
@login_required
def export_schedule_csv(exam_id):
    exam = get_exam_or_none(exam_id, current_user.id)
    if not exam:
        flash('Schedule not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))

    db = get_db()
    rows = list(db.schedules.find(
        {'exam_id': exam['_id']},
        sort=[('date', 1), ('slot_start', 1)]
    ))

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Start', 'End', 'Subject', 'Chapter', 'Duration (min)'])
    for r in rows:
        date_val = r['date'] if isinstance(r['date'], str) else r['date'].isoformat()
        cw.writerow([date_val, r['slot_start'], r['slot_end'], r['subject'], r.get('chapter', ''), r['duration_minutes']])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=exam_{exam_id}_schedule.csv"
    output.headers["Content-type"] = "text/csv"
    return output


@schedule_bp.route('/schedule/<exam_id>/export/pdf')
@login_required
def export_schedule_pdf(exam_id):
    exam = get_exam_or_none(exam_id, current_user.id)
    if not exam:
        flash('Schedule not found', 'danger')
        return redirect(url_for('schedule.all_schedules'))

    db = get_db()
    rows = list(db.schedules.find(
        {'exam_id': exam['_id']},
        sort=[('date', 1), ('slot_start', 1)]
    ))

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
        date_key = r['date'] if isinstance(r['date'], str) else r['date'].isoformat()
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