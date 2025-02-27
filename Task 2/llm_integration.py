# llm_integration.py
# Functions for generating responses using LLM integration

from typing import List, Dict, Any, Optional
import os

# Try to import Google Generative AI (fallback to placeholder if not available)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

def setup_llm(api_key: Optional[str] = None) -> bool:
    """
    Set up the LLM with the provided API key.
    
    Args:
        api_key: API key for accessing the LLM service
    
    Returns:
        True if setup was successful, False otherwise
    """
    if not GENAI_AVAILABLE:
        print("Google Generative AI package not available")
        return False
    
    if not api_key:
        print("No API key provided")
        return False
        
    try:
        genai.configure(api_key=api_key)
        
        # Test the API key by listing models
        models = genai.list_models()
        if not models:
            print("No models available with the provided API key")
            return False
            
        # Check if any model supports generateContent
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                print(f"Found suitable model: {model.name}")
                return True
                
        print("No models support generateContent")
        return False
        
    except Exception as e:
        print(f"Error setting up Gemini: {str(e)}")
        return False

def generate_response(query: str, relevant_feedback: List[Dict]) -> str:
    """
    Generate a response to a user query using retrieved feedback and an LLM.
    
    Args:
        query: User query
        relevant_feedback: List of relevant feedback items
    
    Returns:
        Generated response
    """
    # Check if this is a greeting or general query
    greeting_words = ["hi", "hello", "hey", "greetings", "howdy"]
    if query.lower() in greeting_words or len(query.split()) <= 2:
        return """Hello! I'm the RTV Feedback Analysis Chatbot. I can help you understand community feedback about Raising The Village programs.

You can ask me questions like:
- "What do people like about Agriculture programs?"
- "What are common complaints about water access?"
- "Why do communities recommend VSLA programs?"

How can I help you today?"""
    
    # If no relevant feedback was found
    if not relevant_feedback:
        return "I don't have enough information to answer this query based on community feedback. Please try asking about specific RTV programs like Agriculture & Nutrition, WASH, Water, Access to Health, or VSLAs."
    
    # Format the feedback as context for the LLM
    context = "Community feedback:\n"
    for i, item in enumerate(relevant_feedback[:5]):
        context += f"{i+1}. Program: {item['program_name']}, Feedback Type: {item['feedback_type']}\n"
        context += f"   Feedback: {item['feedback_text']}\n"
    
    # Create a prompt for the LLM
    prompt = f"""
    Based on the following community feedback about RTV (Raising The Village) programs, 
    please answer this question: "{query}"
    
    {context}
    
    Please synthesize the feedback to provide an insightful, detailed answer (at least 3-4 sentences). 
    Focus only on what can be determined from the provided feedback. If the feedback doesn't 
    address the query, state that clearly but try to provide related information that might be helpful.
    Format your answer in a clear, conversational manner.
    """
    
    # Call LLM API if available, otherwise return a fallback response
    if GENAI_AVAILABLE:
        try:
            # Try the recommended model first
            recommended_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
            
            for model_name in recommended_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as model_error:
                    print(f"Error with model {model_name}: {str(model_error)}")
                    continue
            
            # If we get here, none of the recommended models worked
            print("All recommended models failed, falling back to dynamic model selection")
            
            # Try to find any suitable model
            models = genai.list_models()
            for model in models:
                if "generateContent" in model.supported_generation_methods:
                    try:
                        model_name = model.name
                        model_instance = genai.GenerativeModel(model_name)
                        response = model_instance.generate_content(prompt)
                        return response.text
                    except Exception as model_error:
                        print(f"Error with model {model_name}: {str(model_error)}")
                        continue
            
            # If we get here, all models failed
            print("No suitable Gemini model found or all models failed")
            return generate_fallback_response(query, relevant_feedback)
                
        except Exception as e:
            # Log the error for debugging
            print(f"Gemini API error: {str(e)}")
            # Fallback to rule-based response if API fails
            return generate_fallback_response(query, relevant_feedback)
    else:
        return generate_fallback_response(query, relevant_feedback)

def generate_fallback_response(query: str, relevant_feedback: List[Dict]) -> str:
    """Generate a simple response based on the retrieved feedback without using an LLM."""
    # Count positive vs negative feedback
    positive_count = sum(1 for item in relevant_feedback if item['feedback_type'] == 'positive')
    negative_count = sum(1 for item in relevant_feedback if item['feedback_type'] == 'negative')
    total = len(relevant_feedback)
    
    # Get the most mentioned program
    programs = {}
    for item in relevant_feedback:
        prog = item['program_name']
        programs[prog] = programs.get(prog, 0) + 1
    most_mentioned = max(programs.items(), key=lambda x: x[1])[0] if programs else "unknown"
    
    # Generate a simple response
    response = f"Based on {total} community feedback items about {most_mentioned} program:\n\n"
    
    if positive_count > negative_count:
        response += f"The feedback is generally positive ({positive_count}/{total} positive comments).\n\n"
    elif negative_count > positive_count:
        response += f"The feedback is generally negative ({negative_count}/{total} negative comments).\n\n"
    else:
        response += f"The feedback is mixed ({positive_count} positive and {negative_count} negative comments).\n\n"
    
    # Add a few example feedback items
    response += "Example feedback:\n"
    for i, item in enumerate(relevant_feedback[:3]):
        response += f"- {item['feedback_text']}\n"
    
    return response 

def list_available_models():
    """List all available models from the Gemini API with detailed information."""
    if GENAI_AVAILABLE:
        try:
            models = genai.list_models()
            model_info = []
            for model in models:
                model_info.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "supported_generation_methods": model.supported_generation_methods
                })
            return model_info
        except Exception as e:
            return f"Error listing models: {str(e)}"
    return [] 