import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file automatically in local dev

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-fallback-key-change-in-production')
    MONGO_URI = os.environ.get('MONGO_URI', '')
    WTF_CSRF_ENABLED = True

    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)