# retrieval.py
# Functions for retrieving and analyzing feedback

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from preprocessing import preprocess_text

def search_similar_feedback(query: str, 
                           vectorizer: TfidfVectorizer, 
                           index: faiss.Index, 
                           metadata: List[Dict], 
                           top_k: int = 5) -> List[Dict]:
    """
    Search for feedback similar to the given query.
    
    Args:
        query: The search query
        vectorizer: Fitted TF-IDF vectorizer
        index: FAISS index
        metadata: Feedback metadata
        top_k: Number of results to return
    
    Returns:
        List of dictionaries containing matched feedback
    """
    # Preprocess the query
    query_preprocessed = preprocess_text(query)
    
    # Vectorize the query
    query_vector = vectorizer.transform([query_preprocessed]).toarray().astype('float32')
    
    # Search the index
    distances, indices = index.search(query_vector, top_k)
    
    # Get the results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result['distance'] = float(distances[0][i])
            result['similarity'] = 1 / (1 + result['distance'])  # Convert distance to similarity
            results.append(result)
    
    return results

def search_by_program(program_name: str, 
                     metadata: List[Dict], 
                     feedback_type: Optional[str] = None, 
                     top_k: int = 10) -> List[Dict]:
    """
    Search for feedback related to a specific program, optionally filtering by feedback type.
    
    Args:
        program_name: Name of the program (e.g., "Agriculture & Nutrition")
        metadata: Feedback metadata
        feedback_type: Optional filter for 'positive' or 'negative' feedback
        top_k: Maximum number of results to return
    
    Returns:
        List of matching feedback items
    """
    # Filter metadata by program and feedback type
    filtered_metadata = metadata.copy()
    
    # Filter by program name
    filtered_metadata = [item for item in filtered_metadata 
                        if program_name.lower() in item['program_name'].lower()]
    
    # Optionally filter by feedback type
    if feedback_type:
        filtered_metadata = [item for item in filtered_metadata 
                           if item['feedback_type'] == feedback_type]
    
    # Sort by relevance (index as proxy)
    filtered_metadata = sorted(filtered_metadata, key=lambda x: x['index'])
    
    # Return the top k results
    return filtered_metadata[:top_k]

def aggregate_feedback_by_program(metadata: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate feedback statistics by program.
    
    Args:
        metadata: Feedback metadata
    
    Returns:
        Dictionary with program statistics
    """
    result = {}
    
    for program_name in set(item['program_name'] for item in metadata):
        if program_name == 'None':
            continue
            
        program_feedback = [item for item in metadata if item['program_name'] == program_name]
        positive_feedback = [item for item in program_feedback if item['feedback_type'] == 'positive']
        negative_feedback = [item for item in program_feedback if item['feedback_type'] == 'negative']
        
        result[program_name] = {
            'total_feedback': len(program_feedback),
            'positive_feedback': len(positive_feedback),
            'negative_feedback': len(negative_feedback),
            'positive_percentage': len(positive_feedback) / len(program_feedback) * 100 if program_feedback else 0,
            'negative_percentage': len(negative_feedback) / len(program_feedback) * 100 if program_feedback else 0
        }
    
    return result 