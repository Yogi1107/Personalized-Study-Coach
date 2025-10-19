"""
llm_service.py

Hybrid LLM service:
- Phase 2: Single-note AI assistance (summarize, questions, explain)
- Phase 4: Cross-note RAG-based contextual AI (all uploaded notes)
"""

import os
import traceback

# -------------------- Offline Libraries -------------------- #
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# -------------------- Gemini (Online) -------------------- #
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        ONLINE_MODE = True
    else:
        ONLINE_MODE = False
except Exception:
    ONLINE_MODE = False

# -------------------- Phase 4 RAG / TF-IDF -------------------- #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Global storage for notes
notes_index = []  # [{'id': 1, 'text': '...'}]
notes_texts = []
tfidf_vectorizer = TfidfVectorizer()

# -------------------- Phase 4: RAG Functions -------------------- #
def add_note_to_index(note_id, text):
    """
    Add a note to the RAG index (TF-IDF) for cross-note retrieval.
    """
    global notes_index, notes_texts, tfidf_vectorizer
    notes_index.append({'id': note_id, 'text': text})
    notes_texts.append(text)
    tfidf_vectorizer.fit(notes_texts)  # Refit after adding a new note

def rag_answer(query):
    """
    Retrieve relevant note content using TF-IDF cosine similarity across all notes.
    """
    if not notes_texts:
        return "No notes uploaded yet."
    query_vec = tfidf_vectorizer.transform([query])
    notes_vecs = tfidf_vectorizer.transform(notes_texts)
    similarities = cosine_similarity(query_vec, notes_vecs)[0]
    best_idx = np.argmax(similarities)
    best_note = notes_index[best_idx]
    return f"Based on note #{best_note['id']}:\n{best_note['text'][:500]}..."  # preview first 500 chars

# -------------------- Helper Functions -------------------- #
def chunk_text(text, max_chars=1500):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos != -1:
                end = newline_pos
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# -------------------- Phase 2: Single-Note Functions -------------------- #
def summarize_text(text, sentences_count=5):
    """
    Summarize a single note (Phase 2)
    """
    if ONLINE_MODE:
        try:
            response = genai.generate_text(
                model="gemini-1.5",
                prompt=f"Summarize the following text in simple terms:\n\n{text}",
                max_output_tokens=300
            )
            return response.result if hasattr(response, "result") else str(response)
        except Exception:
            traceback.print_exc()
            pass

    # Offline fallback
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return ". ".join(sentences[:sentences_count])

def generate_questions_from_text(text, num_questions=5):
    """
    Generate practice questions from a single note (Phase 2)
    """
    if ONLINE_MODE:
        try:
            response = genai.generate_text(
                model="gemini-1.5",
                prompt=f"Generate {num_questions} practice questions from the following text:\n{text}",
                max_output_tokens=300
            )
            return response.result if hasattr(response, "result") else str(response)
        except Exception:
            traceback.print_exc()
            pass

    # Offline fallback
    chunks = chunk_text(text)
    questions = []
    for chunk in chunks:
        sentences = [s.strip() for s in chunk.split('.') if s.strip()]
        for i, sentence in enumerate(sentences[:num_questions]):
            questions.append(f"What is: {sentence}?")
    return "\n\n".join(questions)

def explain_topic(text, user_question):
    """
    Explain a topic from a single note (Phase 2)
    """
    if ONLINE_MODE:
        try:
            response = genai.generate_text(
                model="gemini-1.5",
                prompt=f"Based on the following text:\n{text}\n\nExplain the question in simple terms:\n{user_question}",
                max_output_tokens=300
            )
            return response.result if hasattr(response, "result") else str(response)
        except Exception:
            traceback.print_exc()
            pass

    # Offline fallback
    chunks = chunk_text(text)
    explanations = []
    for chunk in chunks:
        sentences = [s.strip() for s in chunk.split('.') if s.strip()]
        relevant = [s for s in sentences if any(word.lower() in s.lower() for word in user_question.split())]
        explanations.append(" ".join(relevant) if relevant else "No direct match found, please refer to the text.")
    return "\n\n".join(explanations)

# -------------------- Example Usage -------------------- #
if __name__ == "__main__":
    sample_text = "Python is a high-level programming language. It is widely used for web development, data analysis, AI, and more."
    
    # Phase 2
    print("Phase 2 - Single Note:")
    print("Summary:")
    print(summarize_text(sample_text))
    print("\nQuestions:")
    print(generate_questions_from_text(sample_text))
    print("\nExplanation for 'AI':")
    print(explain_topic(sample_text, "AI"))

    # Phase 4
    add_note_to_index(1, sample_text)
    print("\nPhase 4 - RAG answer for 'web development':")
    print(rag_answer("web development"))
