import psycopg2
from psycopg2.extras import RealDictCursor
from config import Config

# ===================== Database Helper ===================== #

def get_db_connection():
    """
    Returns a psycopg2 connection with RealDictCursor.
    Use as:
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
    """
    conn = psycopg2.connect(Config.DATABASE_URL)
    return conn


def init_db():
    """Initialize PostgreSQL database tables"""
    create_users = '''
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(150) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    );
    '''

    create_notes = '''
    CREATE TABLE IF NOT EXISTS notes (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        title VARCHAR(255) NOT NULL,
        content TEXT NOT NULL,
        file_type VARCHAR(50) NOT NULL,
        upload_date TIMESTAMP NOT NULL,
        summary TEXT,
        questions TEXT,
        is_completed BOOLEAN NOT NULL DEFAULT FALSE
    );
    '''

    # FIX: added user_id to exams so schedules are user-scoped
    create_exams = '''
    CREATE TABLE IF NOT EXISTS exams (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        exam_date DATE NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    '''

    create_subjects = '''
    CREATE TABLE IF NOT EXISTS subjects (
        id SERIAL PRIMARY KEY,
        exam_id INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
        subject_name VARCHAR(255) NOT NULL,
        chapters TEXT,
        priority VARCHAR(50)
    );
    '''

    create_schedules = '''
    CREATE TABLE IF NOT EXISTS schedules (
        id SERIAL PRIMARY KEY,
        exam_id INTEGER REFERENCES exams(id) ON DELETE CASCADE,
        date DATE NOT NULL,
        slot_start TIME NOT NULL,
        slot_end TIME NOT NULL,
        subject VARCHAR(255) NOT NULL,
        chapter TEXT,
        duration_minutes INTEGER NOT NULL,
        created_by VARCHAR(50) NOT NULL DEFAULT 'auto',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    '''

    # Safe migrations for existing databases
    migrate_notes_completed = '''
    ALTER TABLE notes ADD COLUMN IF NOT EXISTS is_completed BOOLEAN NOT NULL DEFAULT FALSE;
    '''

    migrate_exams_user_id = '''
    ALTER TABLE exams ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
    '''

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_users)
            cur.execute(create_notes)
            cur.execute(create_exams)
            cur.execute(create_subjects)
            cur.execute(create_schedules)
            cur.execute(migrate_notes_completed)
            cur.execute(migrate_exams_user_id)