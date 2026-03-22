import os
from dotenv import load_dotenv

# This line reads your .env file and loads the API key into the environment
load_dotenv()

# LLM settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"  # Free model on Groq, fast and good

# Chunking settings
CHUNK_SIZE = 500      # Each chunk is 500 tokens
CHUNK_OVERLAP = 100   # 100 tokens overlap between chunks so context isn't lost
HF_TOKEN = os.getenv("HF_TOKEN")
# Retrieval settings
TOP_K = 5  # How many chunks to retrieve per question

# Embedding model (runs locally, no API cost)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector database location
CHROMA_DB_PATH = "./chroma_db"