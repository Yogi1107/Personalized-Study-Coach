from flask_login import UserMixin
from database import get_db_connection
from psycopg2.extras import RealDictCursor

# ===================== Flask-Login User Class ===================== #

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = str(id)  # flask-login expects id to be str-like
        self.username = username
        self.password = password

    @staticmethod
    def load_user(user_id):
        """
        user_id will be a string; convert to int for DB lookup
        """
        try:
            uid = int(user_id)
        except Exception:
            return None
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT * FROM users WHERE id = %s', (uid,))
                row = cur.fetchone()
                if row:
                    return User(row['id'], row['username'], row['password'])
        return None