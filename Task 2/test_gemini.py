#!/usr/bin/env python3
# test_gemini.py - Test script for Gemini API

import os
import sys
import json

# Set API key
api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyD9AI81Jfkkun70HSYyx-b8jOoaAyGkIVY')

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    # List available models
    print("Listing available models...")
    models = genai.list_models()
    
    print("\nAvailable models:")
    for model in models:
        print(f"- Name: {model.name}")
        print(f"  Display name: {model.display_name}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        print()
    
    # Try specific recommended models
    recommended_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    for model_name in recommended_models:
        try:
            print(f"\nTesting recommended model: {model_name}")
            model_instance = genai.GenerativeModel(model_name)
            response = model_instance.generate_content("Hello, what can you do?")
            print(f"Response: {response.text[:100]}...")
            print(f"Success with model: {model_name}")
            break
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
    else:
        print("\nAll recommended models failed, trying any available model...")
        
        # Try any model that supports generateContent
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                try:
                    print(f"Testing model: {model.name}")
                    model_instance = genai.GenerativeModel(model.name)
                    response = model_instance.generate_content("Hello, what can you do?")
                    print(f"Response: {response.text[:100]}...")
                    print(f"Success with model: {model.name}")
                    break
                except Exception as e:
                    print(f"Error with model {model.name}: {str(e)}")
        else:
            print("No models found that work with generateContent")
    
except ImportError:
    print("Error: google.generativeai package not installed")
    print("Install with: pip install google-generativeai")
    sys.exit(1)
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1) 