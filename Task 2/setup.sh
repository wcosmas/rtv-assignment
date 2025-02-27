#!/bin/bash
# Setup script for RTV Feedback Analysis Chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data
mkdir -p embeddings
mkdir -p models

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

echo "Setup complete! You can now run the chatbot with:"
echo "python rtv_chatbot.py  # For command-line interface"
echo "streamlit run app.py   # For web interface" 