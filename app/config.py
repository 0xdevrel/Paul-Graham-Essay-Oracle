import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "pg_essays.db"

# Model configuration
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_DIMENSION = 3072

# Scraping & Chunking configuration
SCRAPE_INDEX_URL = "https://paulgraham.com/articles.html"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# API Key Validation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_api_key():
    """Returns the Gemini API Key or raises ValueError if not configured."""
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set in the environment or .env file.")
    return key
