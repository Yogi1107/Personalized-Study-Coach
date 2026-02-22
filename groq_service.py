"""
groq_service.py

Hybrid LLM service (Groq + TF-IDF RAG)
"""

import os
import re
import traceback
import numpy as np
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

notes_index = []
notes_texts = []
tfidf_vectorizer = TfidfVectorizer()


def clean_text(response_text: str) -> str:
    if not response_text:
        return ""
    text = str(response_text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def add_note_to_index(note_id, text):
    """Add a note to the in-memory TF-IDF index."""
    global notes_index, notes_texts, tfidf_vectorizer
    notes_index.append({"id": note_id, "text": text})
    notes_texts.append(text)
    tfidf_vectorizer.fit(notes_texts)


# FIX: added user_id parameter to match rag_service.py call signature
def rag_answer(query, user_id=None):
    """Return an answer based on the most relevant indexed note using TF-IDF."""
    if not notes_texts:
        return "No notes uploaded yet."

    try:
        query_vec = tfidf_vectorizer.transform([query])
        notes_vecs = tfidf_vectorizer.transform(notes_texts)
        similarities = cosine_similarity(query_vec, notes_vecs)[0]
        best_idx = np.argmax(similarities)
        best_note = notes_index[best_idx]
        context = best_note["text"][:3000]

        prompt = f"""You are a helpful study assistant.

Context (from note #{best_note['id']}):
{context}

Question: {query}

Provide a clear, educational answer based on the context above."""

        response = groq_generate(prompt)
        if response and not response.startswith("Error"):
            return f"Based on note #{best_note['id']}:\n\n{response}"
        return f"Error generating answer: {response}"

    except Exception as e:
        traceback.print_exc()
        return f"Error during RAG: {str(e)}"


def chunk_text(text, max_chars=3000):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos != -1:
                end = newline_pos
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def groq_generate(prompt, temperature=0.7, max_tokens=2048, retry_count=0):
    """Generate text from Groq with error handling."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant. Provide clear, well-structured answers."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        if not response or not response.choices:
            return "Error: No response from Groq API"

        text = response.choices[0].message.content
        if not text or len(text.strip()) == 0:
            return "Error: Empty response from Groq"

        return clean_text(text)

    except Exception as e:
        error_msg = str(e)
        if retry_count < 1:
            return groq_generate(prompt, temperature, max_tokens, retry_count + 1)
        traceback.print_exc()
        return f"Error generating response from Groq: {error_msg}"


def summarize_text(text, sentences_count=5):
    if len(text) > 3000:
        text = text[:3000] + "..."

    prompt = f"""Summarize the following text in approximately {sentences_count} clear, concise sentences.

Text:
{text}

Summary:"""

    response = groq_generate(prompt)
    if response and not response.startswith("Error"):
        return response
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    return " ".join(sentences[:sentences_count])


def generate_questions_from_text(text, num_questions=5):
    if len(text) > 3000:
        text = text[:3000] + "..."

    prompt = f"""Based on the following study note, generate exactly {num_questions} clear, thoughtful questions that test understanding of the key concepts.

Format: One question per line, numbered 1-{num_questions}.

Note:
{text}

Questions:"""

    response = groq_generate(prompt)
    questions = []

    if response and not response.startswith("Error"):
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        for line in lines:
            clean_line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if clean_line and len(clean_line) > 10:
                questions.append({"text": clean_line, "completed": False})
                if len(questions) >= num_questions:
                    break

    if not questions:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        for sentence in sentences[:num_questions]:
            questions.append({
                "text": f"What does this mean: {sentence[:100]}?",
                "completed": False
            })

    return questions[:num_questions]


def explain_topic(note_content: str, user_question: str) -> str:
    if len(note_content) > 3000:
        note_content = note_content[:3000] + "..."

    prompt = f"""Based on the following note, please explain the concept clearly and simply.

Note:
{note_content}

Question: {user_question}

Explanation:"""

    response = groq_generate(prompt)
    if response and not response.startswith("Error"):
        return response
    return f"Error: Unable to generate explanation. {response}"