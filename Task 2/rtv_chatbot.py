# rtv_chatbot.py
# Main module for the RTV Feedback Analysis Chatbot

import os
import sys
import pandas as pd
from typing import Dict, List, Any, Optional

# Import modules
from config import (
    FEEDBACK_DATA_PATH, VECTORIZER_PATH, FAISS_INDEX_PATH, 
    METADATA_PATH, PROGRAM_CODES, GEMINI_API_KEY
)
from preprocessing import process_feedback_data, preprocess_text
from embeddings import (
    create_embeddings, build_faiss_index, create_metadata, 
    save_embeddings_artifacts, load_embeddings_artifacts
)
from retrieval import search_similar_feedback, search_by_program, aggregate_feedback_by_program
from llm_integration import setup_llm, generate_response

class RTVChatbot:
    """Main chatbot class for RTV feedback analysis."""
    
    def __init__(self, load_existing: bool = True):
        """
        Initialize the chatbot.
        
        Args:
            load_existing: Whether to load existing artifacts or create new ones
        """
        self.vectorizer = None
        self.index = None
        self.metadata = None
        
        # Set up LLM
        self.llm_available = setup_llm(GEMINI_API_KEY)
        
        # Load or create artifacts
        if load_existing and os.path.exists(VECTORIZER_PATH) and os.path.exists(FAISS_INDEX_PATH):
            self._load_artifacts()
        elif os.path.exists(FEEDBACK_DATA_PATH):
            self._create_artifacts()
        else:
            raise FileNotFoundError(f"No feedback data found at {FEEDBACK_DATA_PATH}")
    
    def _load_artifacts(self) -> None:
        """Load existing embeddings artifacts."""
        paths = {
            'vectorizer': VECTORIZER_PATH,
            'index': FAISS_INDEX_PATH,
            'metadata': METADATA_PATH
        }
        self.vectorizer, self.index, self.metadata = load_embeddings_artifacts(paths)
        print(f"Loaded {len(self.metadata)} feedback entries from existing artifacts")
    
    def _create_artifacts(self) -> None:
        """Create new embeddings artifacts from feedback data."""
        # Load and process feedback data
        df = pd.read_csv(FEEDBACK_DATA_PATH)
        feedback_df = process_feedback_data(df, PROGRAM_CODES)
        
        # Create embeddings and index
        embeddings, self.vectorizer = create_embeddings(feedback_df)
        self.index = build_faiss_index(embeddings)
        self.metadata = create_metadata(feedback_df)
        
        # Save artifacts
        paths = {
            'vectorizer': VECTORIZER_PATH,
            'embeddings': os.path.join(os.path.dirname(FAISS_INDEX_PATH), 'embeddings.npy'),
            'index': FAISS_INDEX_PATH,
            'metadata': METADATA_PATH
        }
        save_embeddings_artifacts(embeddings, self.vectorizer, self.index, self.metadata, paths)
        print(f"Created and saved artifacts for {len(self.metadata)} feedback entries")
    
    def process_query(self, query: str) -> Dict:
        """
        Process a query and generate a response.
        
        Args:
            query: User query
        
        Returns:
            Dictionary with response and retrieved feedback
        """
        # Search for relevant feedback
        relevant_feedback = search_similar_feedback(
            query, self.vectorizer, self.index, self.metadata
        )
        
        # Generate a response
        response = generate_response(query, relevant_feedback)
        
        return {
            'query': query,
            'response': response,
            'retrieved_feedback': relevant_feedback
        }
    
    def get_program_feedback(self, program_name: str, feedback_type: Optional[str] = None) -> List[Dict]:
        """
        Get feedback for a specific program.
        
        Args:
            program_name: Name of the program
            feedback_type: Optional filter for 'positive' or 'negative' feedback
        
        Returns:
            List of feedback items
        """
        return search_by_program(program_name, self.metadata, feedback_type)
    
    def get_program_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all programs.
        
        Returns:
            Dictionary with program statistics
        """
        return aggregate_feedback_by_program(self.metadata)

    def test_llm_connection(self):
        """Test the connection to the LLM API and return diagnostic information."""
        if not self.llm_available:
            return "LLM integration is not available. Check if the required packages are installed."
        
        from llm_integration import list_available_models
        
        try:
            models = list_available_models()
            return {
                "status": "connected",
                "available_models": models
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e)
            }

def main():
    """Main function for command-line usage."""
    try:
        chatbot = RTVChatbot()
        
        # Simple command-line interface
        print("RTV Feedback Analysis Chatbot")
        print("Type 'exit' to quit")
        print("Type 'stats' to see program statistics")
        print("Type 'program:Program Name' to see feedback for a specific program")
        
        while True:
            query = input("\nQuestion: ")
            
            if query.lower() == 'exit':
                break
            elif query.lower() == 'stats':
                stats = chatbot.get_program_statistics()
                for program, data in stats.items():
                    print(f"\n{program}:")
                    print(f"  Total feedback: {data['total_feedback']}")
                    print(f"  Positive: {data['positive_feedback']} ({data['positive_percentage']:.1f}%)")
                    print(f"  Negative: {data['negative_feedback']} ({data['negative_percentage']:.1f}%)")
            elif query.lower().startswith('program:'):
                program = query[8:].strip()
                feedback = chatbot.get_program_feedback(program)
                print(f"\nFeedback for {program}:")
                for i, item in enumerate(feedback[:5]):
                    print(f"  {i+1}. [{item['feedback_type']}] {item['feedback_text']}")
            else:
                result = chatbot.process_query(query)
                print("\nResponse:")
                print(result['response'])
                
                print("\nBased on:")
                for i, item in enumerate(result['retrieved_feedback'][:3]):
                    print(f"  {i+1}. [{item['program_name']} - {item['feedback_type']}] {item['feedback_text'][:100]}...")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 