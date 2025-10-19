import re
from difflib import SequenceMatcher
from llm_service import gemini_generate  # make sure llm_service.py is in the same folder


def clean_text(text):
    """Remove special characters, emojis, and extra whitespace."""
    # Remove non-alphanumeric (keep spaces, commas, periods)
    text = re.sub(r'[^A-Za-z0-9.,?;:()\'" \n]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_relevant_chunks(note_text, query, num_chunks=3):
    """
    Simple retrieval: split the note text and return the most relevant chunks.
    Uses SequenceMatcher for similarity scoring.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]', note_text) if len(s.strip()) > 20]
    if not sentences:
        return [note_text]

    scored = []
    for sent in sentences:
        score = SequenceMatcher(None, sent.lower(), query.lower()).ratio()
        scored.append((score, sent))

    scored.sort(reverse=True)
    top_chunks = [s for _, s in scored[:num_chunks]]
    return top_chunks


def answer_with_context(note_text, query):
    """
    RAG pipeline:
    1. Retrieve relevant note chunks.
    2. Ask Gemini to generate a clean, short, accurate answer.
    3. Clean and return text output (no special symbols).
    """
    relevant_chunks = find_relevant_chunks(note_text, query)
    context = "\n".join(relevant_chunks)

    prompt = f"""
You are an AI tutor. Use only the information from the context below to answer accurately.
Keep the answer short, factual, and clear. Avoid special characters.

Context:
{context}

Question:
{query}

Answer:
"""
    try:
        response = gemini_generate(prompt)
        if not response:
            # Retry once if Gemini stops early (finish_reason=2)
            response = gemini_generate(prompt)
        if response:
            return clean_text(response)
        else:
            return "Sorry, I couldn't generate a reliable answer."
    except Exception as e:
        print("RAG Error:", e)
        return "Sorry, something went wrong while generating the answer."


def generate_summary(note_text):
    """
    Generate a concise summary of a note.
    """
    prompt = f"Summarize this text in 4-5 lines clearly and simply:\n\n{note_text}"
    try:
        response = gemini_generate(prompt)
        if response:
            return clean_text(response)
        else:
            return "Summary unavailable."
    except Exception as e:
        print("Summary Error:", e)
        return "Error while summarizing the note."
