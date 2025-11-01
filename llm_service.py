"""
llm_service.py

Hybrid LLM service (Gemini version):
- Phase 2: Single-note AI assistance (summarize, questions, explain)
- Phase 4: Cross-note RAG-based contextual AI (all uploaded notes)
"""

import os
import re
import traceback
import numpy as np
from dotenv import load_dotenv

# -------------------- Gemini Setup -------------------- #
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# You can switch models here
MODEL_NAME = "gemini-2.5-pro"  # or "gemini-1.5-flash"

# -------------------- Offline RAG Libraries -------------------- #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

notes_index = []
notes_texts = []
tfidf_vectorizer = TfidfVectorizer()

# -------------------- Helper: Clean Text -------------------- #
def clean_text(response_text: str) -> str:
    """
    Removes markdown, bullets, emojis, and special characters.
    Keeps plain sentences only.
    """
    if not response_text:
        return ""
    text = str(response_text)

    # Remove markdown and list markers
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[\-\*\•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)

    # Remove emojis and non-text symbols
    text = re.sub(r"[^\w\s,.?!]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# -------------------- Note Indexing -------------------- #
def add_note_to_index(note_id, text):
    """Add a note to the in-memory RAG index."""
    global notes_index, notes_texts, tfidf_vectorizer
    cleaned_text = clean_text(text)
    notes_index.append({'id': note_id, 'text': cleaned_text})
    notes_texts.append(cleaned_text)
    tfidf_vectorizer.fit(notes_texts)

# -------------------- RAG Retrieval -------------------- #
def rag_answer(query):
    """Return an answer based on the most relevant note using TF-IDF."""
    if not notes_texts:
        return "No notes uploaded yet."

    try:
        cleaned_query = clean_text(query)
        query_vec = tfidf_vectorizer.transform([cleaned_query])
        notes_vecs = tfidf_vectorizer.transform(notes_texts)
        similarities = cosine_similarity(query_vec, notes_vecs)[0]
        best_idx = np.argmax(similarities)
        best_note = notes_index[best_idx]
        context = best_note['text'][:1500]

        prompt = f"""
You are a helpful study assistant.
Context (from note #{best_note['id']}):
{context}

Question: {cleaned_query}

Provide a plain text, educational answer based only on the context.
Do not use bullets, markdown, or special characters.
"""

        response = gemini_generate(prompt)
        cleaned_response = clean_text(response)
        return f"Based on note #{best_note['id']}: {cleaned_response}"
    except Exception as e:
        traceback.print_exc()
        return f"Error during RAG: {str(e)}"

# -------------------- Chunk Text -------------------- #
def chunk_text(text, max_chars=1500):
    """Split long text into manageable chunks."""
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

# -------------------- Gemini Generation -------------------- #
def gemini_generate(prompt, temperature=0.6, max_output_tokens=2048, debug=False, retry_count=0):
    """Generate text from Gemini safely with robust error handling."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=0.9,
                top_k=40,
            ),
        )

        if debug:
            print("---- GEMINI DEBUG ----")
            print(response)
            print("----------------------")

        if not hasattr(response, "candidates") or not response.candidates:
            return "No valid response from Gemini."

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)

        # Retry if Gemini stopped early
        if finish_reason == 2 and retry_count < 1:
            return gemini_generate(
                prompt + "\nContinue your answer.",
                temperature,
                max_output_tokens,
                debug,
                retry_count + 1,
            )

        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text")]
            if text_parts:
                return clean_text("\n".join(text_parts))

        if hasattr(response, "text") and response.text:
            return clean_text(response.text)

        return "Gemini returned no usable text."

    except Exception as e:
        traceback.print_exc()
        return f"Error generating response from Gemini: {str(e)}"

# -------------------- Summarize -------------------- #
def summarize_text(text, sentences_count=5):
    """Summarize a given note text into plain sentences."""
    cleaned_text = clean_text(text)
    prompt = f"Summarize the following text in about {sentences_count} short sentences. Avoid markdown or symbols.\n\n{cleaned_text}"
    response = gemini_generate(prompt)
    if response:
        return clean_text(response)

    # Fallback simple summarization
    sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
    return ". ".join(sentences[:sentences_count])

# -------------------- Generate Questions -------------------- #
def generate_questions_from_text(text, num_questions=5):
    """Generate conceptual questions from a study note."""
    cleaned_text = clean_text(text)
    prompt = f"Generate {num_questions} short, one-line conceptual questions based on this study note. Avoid any special symbols.\n\n{cleaned_text}"
    response = gemini_generate(prompt)

    questions = []
    if response:
        lines = [clean_text(line) for line in response.split("\n") if line.strip()]
        for line in lines[:num_questions]:
            questions.append({"text": line, "completed": False})

    if not questions:
        chunks = chunk_text(cleaned_text)
        for chunk in chunks:
            sentences = [s.strip() for s in chunk.split('.') if s.strip()]
            for i, sentence in enumerate(sentences[:num_questions]):
                q = f"What is {sentence}?"
                questions.append({"text": clean_text(q), "completed": False})
            if len(questions) >= num_questions:
                break

    return questions[:num_questions]

# -------------------- Explain Topic -------------------- #
def explain_topic(note_content: str, user_question: str) -> str:
    """Explain a concept in plain text without symbols."""
    cleaned_note = clean_text(note_content)
    cleaned_question = clean_text(user_question)

    prompt = f"""
You are an AI teaching assistant. Explain clearly and simply based on the given note.
Avoid using markdown, bullets, or any special characters.

Note:
{cleaned_note}

Question:
{cleaned_question}

Provide a short, clear explanation in plain English.
"""
    raw_response = gemini_generate(prompt)
    return clean_text(raw_response)

# -------------------- Example Usage -------------------- #
if __name__ == "__main__":
    sample_text = (
        "Python is a high-level programming language used for AI and data science. "
        "It is simple, readable, and supports multiple paradigms."
    )

    print("Summary:\n", summarize_text(sample_text))
    print("\nQuestions:\n", generate_questions_from_text(sample_text))
    print("\nExplanation:\n", explain_topic(sample_text, "What is AI?"))
    add_note_to_index(1, sample_text)
    print("\nRAG Answer:\n", rag_answer("data science"))