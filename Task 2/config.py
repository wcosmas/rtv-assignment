# config.py
# Configuration settings for the RTV Feedback Analysis System

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
for dir_path in [DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Files
FEEDBACK_DATA_PATH = os.path.join(DATA_DIR, 'processed_feedback.csv')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, 'feedback_index.faiss')
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, 'feedback_metadata.json')

# API Settings
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'your_api_key')  # Replace with your actual key

# Program codes
PROGRAM_CODES = {
    1: "Agriculture & Nutrition",
    2: "WASH",
    3: "Water",
    4: "Access to Health",
    5: "VSLAs",
    99: "None"
}

# Processing settings
MAX_FEATURES = 1000
NGRAM_RANGE = (1, 2)
TOP_K_RESULTS = 5 