# Personalized Study Coach

A web application that helps students study efficiently by combining their course notes with AI-powered learning assistance.

---

## What is it?

Personalized Study Coach allows users to:

- Upload notes (PDFs or text) and extract content automatically.
- Generate AI-powered summaries and practice questions for each note.
- Ask contextual questions about their notes using Retrieval-Augmented Generation (RAG).
- Create personalized study schedules based on exam dates and subjects.
- Track study progress with an interactive dashboard.

---

## Tech Stack

- **Backend:** Flask (Python)
- **Database:** MongoDB (via PyMongo)
- **Frontend:** HTML, CSS, JavaScript, Bootstrap 5
- **AI:** Groq API (summarization, Q&A, explanations)
- **Auth:** Flask-Login with Werkzeug password hashing

---

## Features

1. **Note Management** — Upload, view, and delete PDF or TXT notes with automatic text extraction.
2. **AI Assistance** — Generate summaries, practice questions, and topic explanations powered by Groq.
3. **RAG Chat** — Ask questions across all your uploaded notes with contextual, source-cited answers.
4. **Personalized Study Scheduler** — Day-wise study plan generated from exam dates, subjects, chapters, and priority levels.
5. **Progress Tracking** — Mark notes as complete and monitor your preparation from the dashboard.
6. **Export** — Download your study schedule as CSV or PDF.

---

## Project Structure

```
├── app.py                  # App factory, blueprint registration
├── config.py               # Config (secret key, MongoDB URI, upload folder)
├── database.py             # PyMongo connection + index initialisation
├── models.py               # Flask-Login User class
├── auth.py                 # Register / login / logout routes
├── notes.py                # Note upload, view, delete, complete routes
├── ai.py                   # Summarize, questions, explain, RAG routes
├── schedule.py             # Schedule creation, view, export, delete routes
├── main.py                 # Home dashboard route
├── rag_service.py          # RAG pipeline (chunk retrieval + Groq generation)
├── groq_service.py         # Groq API wrapper
├── utils.py                # Text extraction, scheduling helpers
├── templates/              # Jinja2 HTML templates
└── static/                 # CSS, JS assets
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- MongoDB running locally or a MongoDB Atlas URI
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/personalized-study-coach.git
cd personalized-study-coach

# 2. Create and activate a virtual environment
python -m venv .myenv
# Windows
.myenv\Scripts\activate
# macOS / Linux
source .myenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
# Windows (PowerShell)
$env:SECRET_KEY="your-secret-key"
$env:MONGO_URI="mongodb://localhost:27017/studycoach"
$env:GROQ_API_KEY="your-groq-api-key"

# macOS / Linux
export SECRET_KEY="your-secret-key"
export MONGO_URI="mongodb://localhost:27017/studycoach"
export GROQ_API_KEY="your-groq-api-key"

# 5. Run the app
flask run
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Environment Variables

| Variable | Description |
|---|---|
| `SECRET_KEY` | Flask session secret key |
| `MONGO_URI` | MongoDB connection string |
| `GROQ_API_KEY` | Groq API key for AI features |

---

## Requirements

```
flask
flask-login
pymongo
werkzeug
groq
reportlab
pypdf
python-dotenv
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Deployment

This app is ready to deploy on [Render](https://render.com) or [Railway](https://railway.app):

1. Set the environment variables in your platform's dashboard.
2. Point `MONGO_URI` to a MongoDB Atlas cluster.
3. Set the start command to `flask run --host=0.0.0.0 --port=8000` or use `gunicorn app:app`.

---

## License

MIT
