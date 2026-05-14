"""
rag_service.py

RAG service using sentence-transformers for semantic similarity.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from database import get_db
from groq_service import groq_generate

# Lazy-loaded — won't download until first RAG request
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[^A-Za-z0-9.,?;:()\'" \n]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_relevant_chunks(note_text, query, num_chunks=3):
    """
    Split note into sentences and rank by cosine similarity
    to the query using sentence-transformer embeddings.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]', note_text) if len(s.strip()) > 20]
    if not sentences:
        return [note_text]

    # Encode query and all sentences into embedding vectors
    query_embedding = get_model().encode(query, convert_to_tensor=True)
    sentence_embeddings = get_model().encode(sentences, convert_to_tensor=True)

    # Compute cosine similarity between query and each sentence
    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]

    # Pick top-N sentences by score
    top_indices = scores.argsort(descending=True)[:num_chunks]
    return [sentences[i] for i in top_indices]


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
        if not response or response.startswith("Error"):
            response = groq_generate(prompt, max_tokens=400)

        if response and not response.startswith("Error"):
            return clean_text(response)
        return "Sorry, I couldn't generate a reliable answer."
    except Exception as e:
        print("RAG Error:", e)
        return "Sorry, something went wrong while generating the answer."


def rag_answer(query, user_id=None):
    """RAG answer across all of a user's notes using semantic search."""
    try:
        db = get_db()

        query_filter = {'user_id': user_id} if user_id else {}
        notes = list(db.notes.find(query_filter, {'title': 1, 'content': 1}))

        if not notes:
            return "No notes found in your library. Please upload some notes first."

        # Encode the query once
        query_embedding = get_model().encode(query, convert_to_tensor=True)

        all_chunks = []
        for note in notes:
            sentences = [s.strip() for s in re.split(r'[.!?]', note['content']) if len(s.strip()) > 20]
            if not sentences:
                continue

            sentence_embeddings = get_model().encode(sentences, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, sentence_embeddings)[0]

            # Take top 2 chunks per note
            top_indices = scores.argsort(descending=True)[:2]
            for i in top_indices:
                all_chunks.append({
                    'score': scores[i].item(),
                    'text': sentences[i],
                    'note_title': note['title'],
                    'note_id': str(note['_id'])
                })

        if not all_chunks:
            return "I couldn't find relevant information in your notes to answer this question."

        # Sort all chunks across all notes by score, take top 5
        all_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = all_chunks[:5]

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