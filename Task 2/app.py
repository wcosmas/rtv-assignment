# app.py
# Streamlit web interface for RTV Feedback Analysis Chatbot

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import chatbot modules
from rtv_chatbot import RTVChatbot

# Initialize the chatbot
@st.cache_resource
def get_chatbot():
    return RTVChatbot()

try:
    chatbot = get_chatbot()
    st.title("RTV Community Feedback Chatbot")
    
    st.write("""
    This chatbot analyzes community feedback about Raising The Village (RTV) programs
    and can answer questions about participant experiences.
    """)
    
    # Sidebar for filtering options
    st.sidebar.header("Program Analytics")
    
    # Get program statistics
    program_stats = chatbot.get_program_statistics()
    
    # Create a DataFrame for visualization
    stats_df = pd.DataFrame({
        'Program': list(program_stats.keys()),
        'Positive Feedback (%)': [stats['positive_percentage'] for stats in program_stats.values()],
        'Negative Feedback (%)': [stats['negative_percentage'] for stats in program_stats.values()],
        'Total Feedback': [stats['total_feedback'] for stats in program_stats.values()]
    })
    
    # Display program statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    stats_df.sort_values('Positive Feedback (%)', ascending=False).plot(
        x='Program', 
        y=['Positive Feedback (%)', 'Negative Feedback (%)'],
        kind='bar', 
        stacked=False,
        ax=ax,
        color=['green', 'red']
    )
    plt.title('Feedback Sentiment by Program')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.sidebar.pyplot(fig)
    
    # Program selector
    selected_program = st.sidebar.selectbox(
        "Explore Program Feedback",
        ["All Programs"] + sorted(list(program_stats.keys()))
    )
    
    feedback_type = st.sidebar.radio(
        "Feedback Type",
        ["All", "Positive", "Negative"]
    )
    
    if selected_program != "All Programs":
        st.sidebar.subheader(f"{selected_program} Feedback Examples")
        filter_type = feedback_type.lower() if feedback_type != "All" else None
        program_feedback = chatbot.get_program_feedback(selected_program, filter_type)
        
        for i, feedback in enumerate(program_feedback[:5]):
            st.sidebar.markdown(f"**{i+1}. {feedback['feedback_type'].title()} Feedback:**")
            st.sidebar.write(feedback['feedback_text'])
            st.sidebar.markdown("---")
    
    # Query input
    query = st.text_input("Ask a question about RTV program feedback:", 
                         "What do communities like about Agriculture & Nutrition programs?")
    
    if st.button("Submit"):
        with st.spinner("Analyzing community feedback..."):
            result = chatbot.process_query(query)
            
        st.subheader("Response:")
        st.write(result['response'])
        
        st.subheader("Based on these community feedback items:")
        for i, item in enumerate(result['retrieved_feedback']):
            with st.expander(f"Feedback {i+1} - {item['program_name']} ({item['feedback_type']})"):
                st.write(item['feedback_text'])
                st.caption(f"Similarity: {item['similarity']:.2f}")

except Exception as e:
    st.error(f"Error initializing chatbot: {str(e)}")
    st.info("Please make sure you have the required data files and dependencies installed.") 