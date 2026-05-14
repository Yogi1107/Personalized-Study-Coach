from flask_login import UserMixin
from database import get_db

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = str(id)   # Flask-Login expects a string-like id
        self.username = username
        self.password = password

    @staticmethod
    def load_user(user_id):
        """
        Load a user by their string id.
        MongoDB stores _id as ObjectId; we accept either ObjectId or
        a plain integer id stored in a separate 'id' field, depending
        on your insert strategy. This version uses the _id field.
        """
        from bson import ObjectId
        try:
            oid = ObjectId(user_id)
        except Exception:
            return None

        db = get_db()
        row = db.users.find_one({'_id': oid})
        if row:
            return User(str(row['_id']), row['username'], row['password'])
        return None