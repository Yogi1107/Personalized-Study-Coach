import PyPDF2

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