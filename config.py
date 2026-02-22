import os

class Config:
    # FIX: SECRET_KEY from environment — never hardcode this
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-fallback-key-change-in-production')

    # Database — Render provides this automatically when you attach a PostgreSQL db
    DATABASE_URL = os.environ.get('DATABASE_URL', '')

    # Fix Render's postgres:// prefix — SQLAlchemy and psycopg2 need postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

    # File uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

    @staticmethod
    def init_app(app):
        # Create uploads folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)