import PyPDF2
from itertools import cycle

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
    """
    Assign chapters to schedule slots.
    FIX: Cycles through chapters repeatedly instead of exhausting the list,
    so every slot always has a chapter assigned.
    Also merges slots shorter than MIN_SLOT_MINUTES with the previous slot
    to avoid useless 9-minute fragments.
    """
    MIN_SLOT_MINUTES = 15  # drop slots shorter than this

    # Build cycling iterators per subject (cycle back to start when exhausted)
    chapter_cycles = {}
    for subj, chapters in subject_chapters_map.items():
        if chapters:
            chapter_cycles[subj] = cycle(chapters)
        else:
            chapter_cycles[subj] = cycle([''])

    # Filter out slots that are too short by merging them into previous slot
    merged = []
    for slot in assignments:
        if slot['duration_minutes'] < MIN_SLOT_MINUTES and merged and merged[-1]['subject'] == slot['subject']:
            # Extend previous slot's end time and duration instead
            merged[-1]['slot_end'] = slot['slot_end']
            merged[-1]['duration_minutes'] += slot['duration_minutes']
        else:
            merged.append(slot.copy())

    # Assign chapters using cycling iterators
    results = []
    for slot in merged:
        subj = slot['subject']
        chapter = next(chapter_cycles.get(subj, cycle([''])))
        row = slot.copy()
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