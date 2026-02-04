"""
llm_service.py

Hybrid LLM service (Ollama + Qdrant + HuggingFace embeddings)

- Phase 2: Single-note AI assistance (summarize, questions, explain)
- Phase 4: Cross-note RAG-based contextual AI (all uploaded notes)
"""

import re
import uuid
import traceback
import requests

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ===================== Configuration ===================== #

# Ollama Setup
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

# Embedding Setup
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Qdrant Setup
QDRANT_COLLECTION = "notes_collection"
qdrant = QdrantClient(url="http://localhost:6333")

# Create collection if not exists
if not qdrant.collection_exists(QDRANT_COLLECTION):
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=384,  # MiniLM embedding size
            distance=Distance.COSINE
        )
    )

# ===================== Helper Functions ===================== #

def clean_text(text: str) -> str:
    """Remove markdown, bullets, emojis, and special characters."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"[^\w\s,.?!]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, max_chars=800):
    """Split text into manageable chunks for embedding."""
    text = clean_text(text)
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

# ===================== Ollama Generation ===================== #

def ollama_generate(prompt, temperature=0.6, max_tokens=300):
    """Generate text from Ollama (Gemma 2B)."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            },
            timeout=300
        )
        response.raise_for_status()
        return clean_text(response.json().get("response", ""))

    except Exception as e:
        traceback.print_exc()
        return f"Ollama error: {str(e)}"

# ===================== Vector Database Functions ===================== #

def add_note_to_index(note_id: int, text: str):
    """Add a note to the Qdrant vector database."""
    chunks = chunk_text(text)
    vectors = embedding_model.encode(chunks).tolist()

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "note_id": note_id,
                    "text": chunk
                }
            )
        )

    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )

# ===================== AI Features ===================== #

def summarize_text(text, sentences_count=5):
    """Summarize a given note text."""
    cleaned_text = clean_text(text)[:1500]
    
    prompt = f"""Summarize the following text in about {sentences_count} short sentences.
Avoid markdown, bullets, or special characters.

Text:
{cleaned_text}
"""
    
    response = ollama_generate(prompt, max_tokens=400)
    if response and not response.startswith("Ollama error"):
        return response
    
    # Fallback: return first few sentences
    sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
    return ". ".join(sentences[:sentences_count]) + "."


def generate_questions_from_text(text, num_questions=5):
    """Generate conceptual questions from a study note."""
    cleaned_text = clean_text(text)[:1200]
    
    prompt = f"""Generate {num_questions} short, one-line conceptual questions based on this study note.
Avoid any special symbols.

Note:
{cleaned_text}
"""
    
    response = ollama_generate(prompt, max_tokens=400)
    
    questions = []
    if response and not response.startswith("Ollama error"):
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        for line in lines[:num_questions]:
            if line:
                questions.append({"text": clean_text(line), "completed": False})
    
    # Fallback if no questions generated
    if not questions:
        sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
        for s in sentences[:num_questions]:
            questions.append(
                {"text": clean_text(f"What is {s}?"), "completed": False}
            )
    
    return questions[:num_questions]


def explain_topic(note_content: str, user_question: str) -> str:
    """Explain a concept based on note content."""
    cleaned_note = clean_text(note_content)[:1000]
    cleaned_question = clean_text(user_question)
    
    prompt = f"""Explain clearly and simply using the note.
Use short sentences. No symbols.

Note:
{cleaned_note}

Question:
{cleaned_question}
"""
    
    return ollama_generate(prompt, max_tokens=400)


# ===================== Example Usage ===================== #

if __name__ == "__main__":
    sample_text = (
        "Python is a high-level programming language used for AI and data science. "
        "It is simple, readable, and supports multiple paradigms."
    )

    print("Summary:\n", summarize_text(sample_text))
    print("\nQuestions:\n", generate_questions_from_text(sample_text))
    print("\nExplanation:\n", explain_topic(sample_text, "What is Python used for?"))

    add_note_to_index(1, sample_text)
    print("\nNote added to vector database successfully!")