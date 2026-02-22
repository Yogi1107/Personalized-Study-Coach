"""
rag_service.py

RAG service for answering questions using context from uploaded notes.
"""

import re
from difflib import SequenceMatcher
from database import get_db_connection
from psycopg2.extras import RealDictCursor
from groq_service import groq_generate


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[^A-Za-z0-9.,?;:()\'" \n]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_relevant_chunks(note_text, query, num_chunks=3):
    sentences = [s.strip() for s in re.split(r'[.!?]', note_text) if len(s.strip()) > 20]
    if not sentences:
        return [note_text]

    scored = []
    for sent in sentences:
        score = SequenceMatcher(None, sent.lower(), query.lower()).ratio()
        scored.append((score, sent))

    scored.sort(reverse=True)
    return [s for _, s in scored[:num_chunks]]


def answer_with_context(note_text, query):
    """RAG pipeline for a single note."""
    relevant_chunks = find_relevant_chunks(note_text, query)
    context = "\n".join(relevant_chunks)

    prompt = f"""You are an AI tutor. Use only the information from the context below to answer accurately.
Keep the answer short, factual, and clear.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = groq_generate(prompt, max_tokens=400)
        # FIX: check for "Error" prefix, not "Ollama error"
        if not response or response.startswith("Error"):
            response = groq_generate(prompt, max_tokens=400)

        if response and not response.startswith("Error"):
            return clean_text(response)
        return "Sorry, I couldn't generate a reliable answer."
    except Exception as e:
        print("RAG Error:", e)
        return "Sorry, something went wrong while generating the answer."


def rag_answer(query, user_id=None):
    """RAG answer function that searches across all of a user's notes."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if user_id:
                    cur.execute('SELECT id, title, content FROM notes WHERE user_id = %s', (user_id,))
                else:
                    cur.execute('SELECT id, title, content FROM notes')
                notes = cur.fetchall()

        if not notes:
            return "No notes found in your library. Please upload some notes first."

        all_chunks = []
        for note in notes:
            chunks = find_relevant_chunks(note['content'], query, num_chunks=2)
            for chunk in chunks:
                score = SequenceMatcher(None, chunk.lower(), query.lower()).ratio()
                all_chunks.append({
                    'score': score,
                    'text': chunk,
                    'note_title': note['title'],
                    'note_id': note['id']
                })

        all_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = all_chunks[:5]

        if not top_chunks:
            return "I couldn't find relevant information in your notes to answer this question."

        context_parts = [f"[From: {c['note_title']}]\n{c['text']}" for c in top_chunks]
        context = "\n\n".join(context_parts)

        prompt = f"""You are an AI tutor helping a student. Use the information from the student's notes below to answer their question accurately.
Keep the answer clear, concise, and well-structured.

Context from notes:
{context}

Student's Question:
{query}

Answer:
"""

        response = groq_generate(prompt, max_tokens=500)
        # FIX: check for "Error" prefix, not "Ollama error"
        if not response or response.startswith("Error"):
            response = groq_generate(prompt, max_tokens=500)

        if response and not response.startswith("Error"):
            source_notes = list(set([c['note_title'] for c in top_chunks]))
            answer = clean_text(response)
            if source_notes:
                sources = ", ".join(source_notes[:3])
                answer += f"\n\nSources: {sources}"
            return answer

        return "Sorry, I couldn't generate a reliable answer."

    except Exception as e:
        print("RAG Answer Error:", e)
        return f"Sorry, an error occurred while searching your notes: {str(e)}"