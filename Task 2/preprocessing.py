# preprocessing.py
# Text preprocessing functions for RTV Feedback Analysis System

import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from custom_tokenize import safe_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Union, Any, Optional

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text: str, 
                    remove_stopwords: bool = True, 
                    lemmatize: bool = True) -> str:
    """
    Preprocess text by lowercasing, removing punctuation and stopwords,
    and performing lemmatization.
    
    Args:
        text: Input text to preprocess
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to perform lemmatization
    
    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize using the standard punkt tokenizer
    tokens = safe_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def process_feedback_data(df: pd.DataFrame, program_codes: Dict[int, str]) -> pd.DataFrame:
    """
    Process raw feedback data into a structured format for analysis.
    
    Args:
        df: DataFrame containing raw survey data
        program_codes: Dictionary mapping program codes to names
    
    Returns:
        Processed feedback DataFrame
    """
    feedback_df = pd.DataFrame()
    
    # Process most recommended program feedback
    if 'most_recommend_rtv_program' in df.columns and 'most_recommend_rtv_program_reason' in df.columns:
        most_rec = df[['most_recommend_rtv_program', 'most_recommend_rtv_program_reason']].copy()
        most_rec = most_rec.rename(columns={
            'most_recommend_rtv_program': 'program_code',
            'most_recommend_rtv_program_reason': 'feedback_text'
        })
        most_rec['feedback_type'] = 'positive'
        most_rec['program_name'] = most_rec['program_code'].map(program_codes)
        most_rec['feedback_text_preprocessed'] = most_rec['feedback_text'].apply(preprocess_text)
        feedback_df = pd.concat([feedback_df, most_rec])
    
    # Process least recommended program feedback
    if 'least_recommend_rtv_program' in df.columns and 'least_recommend_rtv_program_reason' in df.columns:
        least_rec = df[['least_recommend_rtv_program', 'least_recommend_rtv_program_reason']].copy()
        least_rec = least_rec.rename(columns={
            'least_recommend_rtv_program': 'program_code',
            'least_recommend_rtv_program_reason': 'feedback_text'
        })
        least_rec['feedback_type'] = 'negative'
        least_rec['program_name'] = least_rec['program_code'].map(program_codes)
        least_rec['feedback_text_preprocessed'] = least_rec['feedback_text'].apply(preprocess_text)
        feedback_df = pd.concat([feedback_df, least_rec])
    
    # Clean and prepare the final feedback dataset
    feedback_df = feedback_df.reset_index(drop=True)
    feedback_df = feedback_df.dropna(subset=['feedback_text'])
    feedback_df = feedback_df[feedback_df['feedback_text'].str.len() > 5]  # Filter out very short feedback
    
    return feedback_df 