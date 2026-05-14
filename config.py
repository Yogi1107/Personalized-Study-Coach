import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-fallback-key-change-in-production')

    # MongoDB URI — e.g. mongodb+srv://user:pass@cluster.mongodb.net/dbname
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/StudyCoach')

    # File uploads
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)