<<<<<<< HEAD
import nltk
nltk.data.find('tokenizers/punkt')
=======
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY not found in environment variables.")

# Configure Gemini
genai.configure(api_key=api_key)

# List all models
models = genai.list_models()

# Display key info safely
for model in models:
    print(f"Model Name: {model.name}")
    print(f"Base Model ID: {getattr(model, 'base_model_id', 'N/A')}")
    print(f"Display Name: {getattr(model, 'display_name', 'N/A')}")
    print(f"Description: {getattr(model, 'description', 'No description available')}")
    print(f"Supported Methods: {getattr(model, 'supported_generation_methods', 'Unknown')}")
    print("-" * 60)
>>>>>>> e537209 (Added updated llm_service, view_note, and bug fixes)
