import re
from difflib import SequenceMatcher
from llm_service import ollama_generate  # make sure llm_service.py is in the same folder
from database import get_db_connection
from psycopg2.extras import RealDictCursor


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
        response = ollama_generate(prompt)
        if not response:
            # Retry once if Gemini stops early (finish_reason=2)
            response = ollama_generate(prompt)
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
        response = ollama_generate(prompt)
        if response:
            return clean_text(response)
        else:
            return "Summary unavailable."
    except Exception as e:
        print("Summary Error:", e)
        return "Error while summarizing the note."


def rag_answer(query, user_id=None):
    """
    RAG answer function that searches across all user's notes.
    
    1. Retrieves all notes from the database (optionally filtered by user_id)
    2. Finds relevant chunks from all notes
    3. Generates an answer using the most relevant context
    
    Args:
        query: The user's question
        user_id: Optional user ID to filter notes (if None, searches all notes)
    
    Returns:
        A clean, factual answer based on the user's notes
    """
    try:
        # Retrieve all notes from database
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if user_id:
                    cur.execute('SELECT id, title, content FROM notes WHERE user_id = %s', (user_id,))
                else:
                    cur.execute('SELECT id, title, content FROM notes')
                notes = cur.fetchall()
        
        if not notes:
            return "No notes found in your library. Please upload some notes first."
        
        # Collect relevant chunks from all notes
        all_chunks = []
        for note in notes:
            note_text = note['content']
            chunks = find_relevant_chunks(note_text, query, num_chunks=2)  # Get top 2 from each note
            for chunk in chunks:
                # Calculate relevance score
                score = SequenceMatcher(None, chunk.lower(), query.lower()).ratio()
                all_chunks.append({
                    'score': score,
                    'text': chunk,
                    'note_title': note['title'],
                    'note_id': note['id']
                })
        
        # Sort all chunks by relevance and take top 5
        all_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = all_chunks[:5]
        
        if not top_chunks:
            return "I couldn't find relevant information in your notes to answer this question."
        
        # Build context with note references
        context_parts = []
        for i, chunk in enumerate(top_chunks, 1):
            context_parts.append(f"[From: {chunk['note_title']}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using Gemini
        prompt = f"""
You are an AI tutor helping a student. Use the information from the student's notes below to answer their question accurately.
Keep the answer clear, concise, and well-structured. Avoid special characters.

Context from notes:
{context}

Student's Question:
{query}

Answer:
"""
        response = ollama_generate(prompt)
        if not response:
            # Retry once if Gemini stops early
            response = ollama_generate(prompt)
        
        if response:
            # Add source references
            source_notes = list(set([chunk['note_title'] for chunk in top_chunks]))
            answer = clean_text(response)
            if len(source_notes) > 0:
                sources = ", ".join(source_notes[:3])  # Show up to 3 sources
                answer += f"\n\nSources: {sources}"
            return answer
        else:
            return "Sorry, I couldn't generate a reliable answer."
    
    except Exception as e:
        print("RAG Answer Error:", e)
        return f"Sorry, an error occurred while searching your notes: {str(e)}"