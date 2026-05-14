from pymongo import MongoClient, ASCENDING
from pymongo.errors import CollectionInvalid
from config import Config

_client = None
_db = None

def get_db():
    """
    Returns the PyMongo database instance (lazy singleton).
    Use as:
        db = get_db()
        db.users.find_one({"username": "alice"})
    """
    global _client, _db
    if _db is None:
        _client = MongoClient(Config.MONGO_URI)
        # Database name is taken from the URI path, or defaults to 'myapp'
        db_name = Config.MONGO_URI.rsplit('/', 1)[-1].split('?')[0] or 'myapp'
        _db = _client[db_name]
    return _db


def init_db():
    """
    Initialize MongoDB collections and indexes.
    MongoDB creates collections implicitly on first insert,
    but we set up indexes here for uniqueness and performance.
    """
    db = get_db()

    # --- users ---
    # unique index on username
    db.users.create_index([('username', ASCENDING)], unique=True)

    # --- notes ---
    # index on user_id for fast per-user queries
    db.notes.create_index([('user_id', ASCENDING)])

    # --- exams ---
    db.exams.create_index([('user_id', ASCENDING)])

    # --- subjects ---
    # subjects reference exam_id
    db.subjects.create_index([('exam_id', ASCENDING)])

    # --- schedules ---
    db.schedules.create_index([('exam_id', ASCENDING)])
    db.schedules.create_index([('date', ASCENDING)])