import os
from pathlib import Path

# for typing
from openai.types.shared_params import Reasoning
from chromadb.config import Settings
from chromadb.api.collection_configuration import CreateCollectionConfiguration


### paths
BASE_DIR = Path(__file__).resolve().parent
BIOGRAPHY_TXT = BASE_DIR / 'data' / 'biography.txt'
CHROMA_PATH = BASE_DIR / 'chromadb'


### OpenAI -- API_KEY in .env
INFERENCE_MODEL = 'gpt-4o'          # Production model; upgrade to gpt-5.x when available
#EMBEDDING_MODEL = 'text-embedding-3-small'  # 1536 dimensions, max 8192 tokens
EMBEDDING_MODEL = 'text-embedding-3-large'   # 3072 dimensions, max 8192 tokens


### ChromaDB
CHROMA_COLLECTION_NAME = 'bio_facts_large'
CHROMA_COLLECTION_CONFIG = CreateCollectionConfiguration(hnsw={"space": "cosine"})
CHROMA_CLIENT_SETTINGS = Settings(anonymized_telemetry=False)  # don't send usage


### RAG retrieval tuning
N_RESULTS = 10
DISTANCE_THRESHOLD = 0.825  # For space=cosine, Chroma uses distance = (1 - cosine_similarity)
                            # Range: 0 (identical) to 2 (opposite); 1 is orthogonal.
NEIGHBOR_WINDOW = 1         # Sliding window: retrieve +/- N adjacent chunks from same section
MAX_CONTEXT_CHUNKS = 15     # Maximum chunks injected into context (after expansion + merge)
MAX_RETAINED_INJECTIONS = 5 # Keep only the N most recent RAG injections in conversation history


### tool processing
MAX_SEQUENTIAL_TOOL_CALLS = 10  # generous; prevent runaway tool recursion


### HF-Spaces deployment:
HUGGINGFACE_DATASET_REPO = 'LSmithPMP/digital-twin-data'  # Private HF dataset for vector store


### 'Pushover' service (for send_notification tool) -- API USER/TOKEN is in .env
PUSHOVER_ENDPOINT = "https://api.pushover.net/1/messages.json"


### logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')


### Security (see security.py for enforcement logic)
MAX_INPUT_LENGTH = 2000              # Maximum characters per user message
MAX_CONVERSATION_TURNS = 50          # Maximum turns before depth limit
MAX_QUERIES_PER_MINUTE = 10          # Per-session query rate limit
MAX_NOTIFICATIONS_PER_HOUR = 5       # Per-session notification rate limit
