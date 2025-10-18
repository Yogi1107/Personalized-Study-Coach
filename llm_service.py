"""
llm_service.py

Lightweight wrapper functions for summarization, question generation,
and explanations using OpenAI API. Works fast and Windows 7 friendly.
"""

import os
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file.")

openai.api_key = OPENAI_API_KEY

# -------------------- Helper Functions -------------------- #

def chunk_text(text, max_chars=1500):
    """
    Split text into chunks of max_chars to avoid token limit issues.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to split at the last newline if possible
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos != -1:
                end = newline_pos
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# -------------------- Main Functions -------------------- #

def summarize_text(text):
    """
    Summarize text using OpenAI API.
    """
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        prompt = f"Summarize the following text in simple terms:\n\n{chunk}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5
        )
        summaries.append(response.choices[0].message.content.strip())
    return "\n\n".join(summaries)


def generate_questions_from_text(text, num_questions=5):
    """
    Generate practice questions from text using OpenAI API.
    """
    chunks = chunk_text(text)
    all_questions = []
    for chunk in chunks:
        prompt = f"Generate {num_questions} practice questions from the following text:\n\n{chunk}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5
        )
        all_questions.append(response.choices[0].message.content.strip())
    return "\n\n".join(all_questions)


def explain_topic(text, user_question):
    """
    Explain a user question based on the note using OpenAI API.
    """
    chunks = chunk_text(text)
    explanations = []
    for chunk in chunks:
        prompt = f"Based on the following text:\n{chunk}\n\nExplain the following question in simple terms:\n{user_question}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5
        )
        explanations.append(response.choices[0].message.content.strip())
    return "\n\n".join(explanations)
