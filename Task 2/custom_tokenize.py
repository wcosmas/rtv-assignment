# custom_tokenize.py
# A safer tokenization function that doesn't rely on punkt_tab

import nltk
from nltk.tokenize import word_tokenize as nltk_word_tokenize

def safe_tokenize(text):
    """
    A wrapper around NLTK's word_tokenize that handles errors gracefully.
    """
    try:
        return nltk_word_tokenize(text)
    except LookupError:
        # Fallback to a simple space-based tokenization if NLTK fails
        return text.split() 