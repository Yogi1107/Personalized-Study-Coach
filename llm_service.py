"""
llm_service.py

Hybrid LLM service (Ollama + Qdrant + HuggingFace embeddings)

- Phase 2: Single-note AI assistance
- Phase 4: Cross-note RAG-based contextual AI
"""

import re
import uuid
import traceback
import requests
import numpy as np

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# -------------------- Ollama Setup -------------------- #
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"

# -------------------- Embedding Setup -------------------- #
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# -------------------- Qdrant Setup -------------------- #
QDRANT_COLLECTION = "notes_collection"

qdrant = QdrantClient(
    url="http://localhost:6333",  # Qdrant running locally or via Docker
)

# Create collection if not exists
if not qdrant.collection_exists(QDRANT_COLLECTION):
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=384,  # MiniLM embedding size
            distance=Distance.COSINE
        )
    )

# -------------------- Helper: Clean Text -------------------- #
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"[^\w\s,.?!]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -------------------- Ollama Generation -------------------- #
def ollama_generate(prompt, temperature=0.6, max_tokens=300):
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

# -------------------- Chunking -------------------- #
def chunk_text(text, max_chars=800):
    text = clean_text(text)
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

# -------------------- Add Note to Vector DB -------------------- #
def add_note_to_index(note_id: int, text: str):
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

# -------------------- RAG Answer -------------------- #
def rag_answer(query: str, top_k=3):
    try:
        query_embedding = embedding_model.encode(query).tolist()

        search_result = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k
        )

        if not search_result:
            return "No relevant notes found."

        context = "\n".join(
            hit.payload["text"][:800] for hit in search_result
        )

        prompt = f"""
You are a helpful study assistant.
Answer using only the provided context.
Use short, simple sentences.
No symbols or bullets.

Context:
{context}

Question:
{clean_text(query)}
"""

        return ollama_generate(prompt)

    except Exception as e:
        traceback.print_exc()
        return f"RAG error: {str(e)}"

# -------------------- Summarize -------------------- #
def summarize_text(text, sentences_count=5):
    cleaned = clean_text(text)[:800]

    prompt = f"""
Summarize the following text in {sentences_count} short sentences.
Use simple language.

Text:
{cleaned}
"""
    return ollama_generate(prompt)

# -------------------- Generate Questions -------------------- #
def generate_questions_from_text(text, num_questions=5):
    cleaned = clean_text(text)[:800]

    prompt = f"""
Generate {num_questions} short one-line questions
from the following text.

Text:
{cleaned}
"""

    response = ollama_generate(prompt)
    lines = [l.strip() for l in response.split("\n") if l.strip()]

    return [
        {"text": clean_text(line), "completed": False}
        for line in lines[:num_questions]
    ]

# -------------------- Explain Topic -------------------- #
def explain_topic(note_content: str, user_question: str):
    cleaned_note = clean_text(note_content)[:800]

    prompt = f"""
Explain clearly using the note below.
Use simple sentences.

Note:
{cleaned_note}

Question:
{clean_text(user_question)}
"""
    return ollama_generate(prompt)

# -------------------- Example Usage -------------------- #
if __name__ == "__main__":
    sample_text = (
        "Python is a high-level programming language widely used "
        "for artificial intelligence and data science. "
        "It is easy to read and supports multiple programming styles."
    )

    add_note_to_index(1, sample_text)

    print("Summary:\n", summarize_text(sample_text))
    print("\nQuestions:\n", generate_questions_from_text(sample_text))
    print("\nExplanation:\n", explain_topic(sample_text, "What is Python used for?"))
    print("\nRAG Answer:\n", rag_answer("Where is Python commonly used?"))