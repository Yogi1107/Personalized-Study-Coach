# ===================== app.py ===================== #

from flask import Flask
from flask_login import LoginManager, current_user
from datetime import datetime
from config import Config
from database import init_db
from models import User

# ===================== Flask App Config ===================== #

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# ===================== Flask-Login Setup ===================== #

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.load_user(user_id)

# ===================== Initialize Database ===================== #

init_db()

# ===================== Register Blueprints ===================== #

# FIX: import directly from module files, not from a 'routes' package
from routes.auth import auth_bp
from routes.main import main_bp
from routes.notes import notes_bp
from routes.ai import ai_bp
from routes.schedule import schedule_bp

app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)
app.register_blueprint(notes_bp)
app.register_blueprint(ai_bp)
app.register_blueprint(schedule_bp)

# ===================== Context Processor ===================== #

@app.context_processor
def inject_now():
    return {'datetime': datetime, 'current_user': current_user}

# ===================== Run App ===================== #

if __name__ == '__main__':
    app.run(debug=True)