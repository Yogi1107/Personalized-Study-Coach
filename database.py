from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import Config
import certifi

_client = None
_db = None

def get_db():
    global _client, _db
    if _db is None:
        try:
            _client = MongoClient(
                Config.MONGO_URI,
                serverSelectionTimeoutMS=5000, # fail fast if Atlas unreachable
                tlsCAFile=certifi.where()
            )
            # Force a connection check
            _client.admin.command('ping')
            db_name = Config.MONGO_URI.rsplit('/', 1)[-1].split('?')[0] or 'studycoach'
            _db = _client[db_name]
            print(f"Connected to MongoDB: {db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"MongoDB connection failed: {e}")
            raise
    return _db


def init_db():
    db = get_db()
    db.users.create_index([('username', ASCENDING)], unique=True)
    db.notes.create_index([('user_id', ASCENDING)])
    db.exams.create_index([('user_id', ASCENDING)])
    db.subjects.create_index([('exam_id', ASCENDING)])
    db.schedules.create_index([('exam_id', ASCENDING)])
    db.schedules.create_index([('date', ASCENDING)])