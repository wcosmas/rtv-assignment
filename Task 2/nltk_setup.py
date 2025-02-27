# nltk_setup.py
# Script to download required NLTK resources

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Print available tokenizers to debug
from nltk.data import path as nltk_data_path
print(f"NLTK data path: {nltk_data_path}")

print("NLTK resources downloaded successfully!")

# Test the tokenizer to ensure it works
from nltk.tokenize import word_tokenize
test_text = "This is a test sentence."
tokens = word_tokenize(test_text)
print(f"Tokenization test: {tokens}") 