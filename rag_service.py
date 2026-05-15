# rag_service.py
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import get_db
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

    # TF-IDF is stateless — no model to load, very low memory
    corpus = sentences + [query]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vec = tfidf_matrix[-1]
    sentence_vecs = tfidf_matrix[:-1]

    scores = cosine_similarity(query_vec, sentence_vecs)[0]
    top_indices = scores.argsort()[::-1][:num_chunks]
    return [sentences[i] for i in top_indices]


def answer_with_context(note_text, query):
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
    try:
        db = get_db()
        query_filter = {'user_id': user_id} if user_id else {}
        notes = list(db.notes.find(query_filter, {'title': 1, 'content': 1}))

        if not notes:
            return "No notes found in your library. Please upload some notes first."

        all_chunks = []
        for note in notes:
            sentences = [s.strip() for s in re.split(r'[.!?]', note['content']) if len(s.strip()) > 20]
            if not sentences:
                continue

            corpus = sentences + [query]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = tfidf_matrix[-1]
            sentence_vecs = tfidf_matrix[:-1]
            scores = cosine_similarity(query_vec, sentence_vecs)[0]

            top_indices = scores.argsort()[::-1][:2]
            for i in top_indices:
                all_chunks.append({
                    'score': float(scores[i]),
                    'text': sentences[i],
                    'note_title': note['title'],
                    'note_id': str(note['_id'])
                })

        if not all_chunks:
            return "I couldn't find relevant information in your notes to answer this question."

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
                answer += f"\n\nSources: {', '.join(source_notes[:3])}"
            return answer

        return "Sorry, I couldn't generate a reliable answer."

    except Exception as e:
        print("RAG Answer Error:", e)
        return f"Sorry, an error occurred while searching your notes: {str(e)}"