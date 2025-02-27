# RTV Feedback Analysis Chatbot

This system analyzes community feedback about Raising The Village (RTV) programs using natural language processing and retrieval-augmented generation techniques.

## Features

- Process and analyze feedback text from survey responses
- Retrieve relevant feedback based on user queries
- Generate coherent, informative responses that synthesize community opinions
- Visualize feedback sentiment by program
- Filter feedback by program and sentiment type

## Setup

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Set up the environment:

   ```
   # On Unix/Linux/Mac
   ./setup.sh

   # On Windows (run in PowerShell)
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Place your feedback data in the `data` directory:

   - The system expects a CSV file with columns for program codes and feedback text
   - Default filename is `processed_feedback.csv` (can be changed in config.py)

4. (Optional) Set up LLM integration:
   - For Google Gemini: Set the environment variable `GEMINI_API_KEY`
   - For other LLMs: Modify the `llm_integration.py` file

## Usage

### Command-line Interface

Run the chatbot from the command line:

```
python rtv_chatbot.py
```

This will start an interactive session where you can:

- Ask questions about program feedback
- Type `stats` to see program statistics
- Type `program:Program Name` to see feedback for a specific program
- Type `exit` to quit

Example questions:

- "What do people like about Agriculture programs?"
- "What are common complaints about water access?"
- "Why do communities recommend VSLA programs?"

### Web Interface

For a more user-friendly experience, run the Streamlit web app:

```
streamlit run app.py
```

This will open a web interface in your browser where you can:

- Ask questions via a text input
- See visualizations of program statistics
- Filter and explore feedback by program and sentiment
- View detailed responses with supporting evidence

## Data Format

The system expects a CSV file with the following columns:

- `most_recommend_rtv_program`: Program code for most recommended program
- `most_recommend_rtv_program_reason`: Text reason for recommendation
- `least_recommend_rtv_program`: Program code for least recommended program
- `least_recommend_rtv_program_reason`: Text reason for not recommending

Program codes:

- 1: Agriculture & Nutrition
- 2: WASH
- 3: Water
- 4: Access to Health
- 5: VSLAs
- 99: None

## System Architecture

The system consists of several components:

- Text preprocessing pipeline
- TF-IDF vectorization for embeddings
- FAISS index for similarity search
- LLM integration for response generation
- Streamlit web interface

## Extending the System

To extend the system:

1. Add new data to the `data` directory
2. Modify `config.py` if your data has different column names
3. Run the chatbot with new data to rebuild the embeddings
4. Customize `llm_integration.py` to use different LLM providers

## How to Run the Chatbot

Here's a step-by-step guide to running the RTV Feedback Analysis Chatbot:

### Prerequisites

- Make sure you have Python 3.7+ installed
- Ensure you have pip installed for package management

### Setting Up the Environment

- **Clone or download the project files** to your local machine
- **Navigate to the project directory** in your terminal or command prompt:
  ```
  cd path/to/Task 2
  ```
- **Set up the environment** using the provided script:

  On Linux/Mac:

  ```
  chmod +x setup.sh
  ./setup.sh
  ```

  On Windows:

  ```
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
  ```

- **Prepare your data**:

  - Place your CSV file with feedback data in the data directory
  - Rename it to processed_feedback.csv or update the path in config.py

- **(Optional) Set up LLM integration**:

  - For Google Gemini:

    ```
    # On Linux/Mac
    export GEMINI_API_KEY=your_api_key_here

    # On Windows
    set GEMINI_API_KEY=your_api_key_here
    ```

### Running the Command-line Interface

- **Activate the virtual environment** (if not already activated):

  ```
  # On Linux/Mac
  source venv/bin/activate

  # On Windows
  venv\Scripts\activate
  ```

- **Run the chatbot**:

  ```
  python rtv_chatbot.py
  ```

- **Interact with the chatbot**:
  - Ask questions like "What do people like about Agriculture programs?"
  - Type `stats` to see program statistics
  - Type `program:Agriculture & Nutrition` to see feedback for a specific program
  - Type `exit` to quit

### Running the Web Interface

- **Activate the virtual environment** (if not already activated)
- **Run the Streamlit app**:

  ```
  streamlit run app.py
  ```

- **Interact with the web interface**:
  - Your browser should automatically open to the app (typically at http://localhost:8501)
  - Use the text input to ask questions
  - Explore the sidebar to see program statistics and filter feedback
  - Click "Submit" to get responses to your queries

### Example Interactions

**Command-line Example:**

```
RTV Feedback Analysis Chatbot
Type 'exit' to quit
Type 'stats' to see program statistics
Type 'program:Program Name' to see feedback for a specific program

Question: What do communities like about Agriculture & Nutrition programs?
Response:
Based on the community feedback, people appreciate Agriculture & Nutrition programs for several reasons:
1. They provide valuable knowledge and skills about modern farming techniques
2. The programs help improve food security and household nutrition
3. Communities mention increased crop yields and better harvests
4. Some feedback highlights the economic benefits through selling surplus produce
5. The programs are seen as sustainable and having long-term impact

Based on:
  1. [Agriculture & Nutrition - positive] We learned better farming methods that have improved our harvests...
  2. [Agriculture & Nutrition - positive] The agriculture program taught us how to grow more food for our families...
  3. [Agriculture & Nutrition - positive] I can now feed my children better food and even sell some crops...

Question: stats
Agriculture & Nutrition:
  Total feedback: 120
  Positive: 98 (81.7%)
  Negative: 22 (18.3%)
WASH:
  Total feedback: 85
  Positive: 62 (72.9%)
  Negative: 23 (27.1%)
Water:
  Total feedback: 95
  Positive: 68 (71.6%)
  Negative: 27 (28.4%)
Access to Health:
  Total feedback: 78
  Positive: 55 (70.5%)
  Negative: 23 (29.5%)
VSLAs:
  Total feedback: 92
  Positive: 76 (82.6%)
  Negative: 16 (17.4%)

Question: exit
```

The web interface provides the same functionality with a more user-friendly experience, including visualizations and expandable feedback items.

## Troubleshooting

If you encounter issues:

- Ensure all dependencies are installed
- Check that the data file exists and has the expected format
- Verify that the NLTK resources were downloaded correctly
- For LLM integration issues, check your API key and internet connection
