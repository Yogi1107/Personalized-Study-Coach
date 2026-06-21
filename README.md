# StudyMitra - Personalized Study Coach

I built a Personalized Study Coach — a full-stack web app to help students study more effectively using AI.

The problem was that students upload notes but rarely revisit them productively, and manually creating a study schedule before exams is tedious.

So I built a Flask backend with MongoDB, where users upload PDF or text notes, and I used the Groq API with a RAG pipeline — TF-IDF retrieval plus LLM generation — to let students ask questions and get summaries grounded in their own notes. I also built a scheduling algorithm that auto-generates a day-wise study plan based on exam dates, subject priority, and available hours.

The result is a fully deployed app on Render with MongoDB Atlas — users can upload notes, chat with their content, and export a personalized schedule as PDF or CSV.

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
- **AI:** Groq API — `llama-3.3-70b-versatile` (summarization, Q&A, explanations, RAG)
- **Auth:** Flask-Login with Werkzeug password hashing

---

## Features

1. **Note Management** — Upload, view, delete, and mark complete PDF or TXT notes with automatic text extraction.
2. **AI Assistance** — Generate summaries, practice questions, and topic explanations powered by Groq.
3. **RAG Chat** — Ask questions across all your uploaded notes with contextual, source-cited answers using TF-IDF retrieval + Groq generation.
4. **Personalized Study Scheduler** — Day-wise study plan generated from exam dates, subjects, chapters, and priority levels (High / Medium / Low).
5. **Progress Tracking** — Mark notes as complete and monitor your preparation from the dashboard.
6. **Export** — Download your study schedule as CSV or PDF (via ReportLab).

---

## Project Structure

```
├── app.py                      # App factory, blueprint registration
├── config.py                   # Config (secret key, MongoDB URI, upload folder)
├── database.py                 # PyMongo connection + index initialisation
├── models.py                   # Flask-Login User class
├── groq_service.py             # Groq API wrapper + in-memory TF-IDF index
├── rag_service.py              # RAG pipeline (MongoDB chunk retrieval + Groq generation)
├── utils.py                    # Text extraction (PDF/TXT), scheduling helpers
├── routes/
│   ├── auth.py                 # Register / login / logout routes
│   ├── main.py                 # Home dashboard route
│   ├── notes.py                # Note upload, view, delete, complete routes
│   ├── ai.py                   # Summarize, questions, explain, RAG routes
│   └── schedule.py             # Schedule creation, view, export, delete routes
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── upload.html
│   ├── notes.html
│   ├── view_note.html
│   ├── summary.html
│   ├── questions.html
│   ├── explain.html
│   ├── rag_chat.html
│   ├── create_schedule.html
│   ├── view_schedule.html
│   └── all_schedules.html
├── static/
│   └── style.css               # Custom styles
├── uploads/                    # Uploaded note files (auto-created)
├── .env                        # Local environment variables (not committed)
└── requirements.txt
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

# 4. Create a .env file in the project root
# (python-dotenv will load it automatically)
```

Create a `.env` file with the following contents:

```env
SECRET_KEY=your-secret-key
MONGO_URI=mongodb://localhost:27017/studycoach
GROQ_API_KEY=your-groq-api-key
```

```bash
# 5. Run the app
flask run
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Environment Variables

| Variable | Description |
|---|---|
| `SECRET_KEY` | Flask session secret key |
| `MONGO_URI` | MongoDB connection string (local or Atlas) |
| `GROQ_API_KEY` | Groq API key for all AI features |

These can be set via a `.env` file in the project root (recommended for local development) or via your platform's environment dashboard for deployment.

---

## Requirements

All dependencies are listed in `requirements.txt`:

```
flask
flask-login
pymongo
werkzeug
groq
reportlab
pypdf
PyPDF2
python-dotenv
gunicorn
certifi
numpy
scikit-learn
sentence-transformers
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Deployment

This app is ready to deploy on [Render](https://render.com) or [Railway](https://railway.app):

1. Create a `.env` equivalent by setting environment variables in your platform's dashboard.
2. Point `MONGO_URI` to a MongoDB Atlas cluster.
3. Set the start command to:

```bash
gunicorn app:app
```

> `flask run` is for development only. Use `gunicorn` in production.

---

## License

MIT
