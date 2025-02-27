# embeddings.py
# Functions for generating and managing embeddings

import numpy as np
import pickle
import faiss
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List, Dict, Any, Optional

def create_embeddings(feedback_df: pd.DataFrame, 
                     max_features: int = 1000, 
                     ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF embeddings from feedback text.
    
    Args:
        feedback_df: DataFrame containing preprocessed feedback
        max_features: Maximum number of features for TF-IDF
        ngram_range: Range of n-grams to consider
    
    Returns:
        Tuple of (embeddings array, fitted vectorizer)
    """
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    # Fit and transform the preprocessed feedback text
    tfidf_matrix = tfidf_vectorizer.fit_transform(feedback_df['feedback_text_preprocessed'])
    
    # Convert to dense array for compatibility
    tfidf_dense = tfidf_matrix.toarray().astype('float32')
    
    return tfidf_dense, tfidf_vectorizer

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for fast similarity search.
    
    Args:
        embeddings: Array of embeddings vectors
    
    Returns:
        FAISS index
    """
    # Get embedding dimension
    dimension = embeddings.shape[1]
    
    # Create index
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    
    # Add embeddings to index
    index.add(embeddings)
    
    return index

def create_metadata(feedback_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Create metadata mapping for embeddings.
    
    Args:
        feedback_df: DataFrame containing feedback data
    
    Returns:
        List of metadata dictionaries
    """
    metadata = []
    for idx, row in feedback_df.iterrows():
        metadata.append({
            'index': idx,
            'program_code': int(row['program_code']),
            'program_name': row['program_name'],
            'feedback_type': row['feedback_type'],
            'feedback_text': row['feedback_text']
        })
    
    return metadata

def save_embeddings_artifacts(embeddings: np.ndarray, 
                              vectorizer: TfidfVectorizer, 
                              index: faiss.Index, 
                              metadata: List[Dict], 
                              paths: Dict[str, str]) -> None:
    """
    Save all embedding artifacts for later use.
    
    Args:
        embeddings: Array of embeddings
        vectorizer: Fitted TF-IDF vectorizer
        index: FAISS index
        metadata: List of metadata dictionaries
        paths: Dictionary of file paths
    """
    # Save vectorizer
    with open(paths['vectorizer'], 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save embeddings
    np.save(paths['embeddings'], embeddings)
    
    # Save FAISS index
    faiss.write_index(index, paths['index'])
    
    # Save metadata
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f)

def load_embeddings_artifacts(paths: Dict[str, str]) -> Tuple[TfidfVectorizer, faiss.Index, List[Dict]]:
    """
    Load embeddings artifacts for use.
    
    Args:
        paths: Dictionary of file paths
    
    Returns:
        Tuple of (vectorizer, index, metadata)
    """
    # Load vectorizer
    with open(paths['vectorizer'], 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load FAISS index
    index = faiss.read_index(paths['index'])
    
    # Load metadata
    with open(paths['metadata'], 'r') as f:
        metadata = json.load(f)
    
    return vectorizer, index, metadata 