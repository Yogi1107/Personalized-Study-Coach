import os

# ===================== Configuration ===================== #

class Config:
    SECRET_KEY = 'your-secret-key-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    DATABASE_URL = "postgresql://postgres:1107@localhost:5432/study_coach"
    
    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)