# Raising The Village (RTV) Projects

This repository contains two projects developed for Raising The Village:

1. **RTV Risk Assessment Model** - A machine learning system to identify households at risk of not achieving the $2/day income target
2. **RTV Feedback Analysis Chatbot** - A natural language processing system to analyze community feedback about RTV programs

## Project 1: RTV Risk Assessment Model

The Risk Assessment Model uses a Random Forest classifier to predict whether a household is at risk (income < $2/day) or not at risk (income >= $2/day) based on various household characteristics.

### Key Features

- Data processing pipeline for cleaning and feature engineering
- Machine learning model for risk prediction
- Monitoring system for model performance and data drift
- Complete MLOps pipeline for orchestration

[View Risk Assessment Model Documentation](Task%201/README.md)

## Project 2: RTV Feedback Analysis Chatbot

The Feedback Analysis Chatbot processes and analyzes community feedback about RTV programs using natural language processing and retrieval-augmented generation techniques.

### Key Features

- Process and analyze feedback text from survey responses
- Retrieve relevant feedback based on user queries
- Generate coherent, informative responses that synthesize community opinions
- Visualize feedback sentiment by program
- Filter feedback by program and sentiment type

[View Feedback Analysis Chatbot Documentation](Task%202/README.md)

## Getting Started

Each project has its own setup instructions and requirements. Please refer to the individual project READMEs for detailed information.

### Prerequisites

- Python 3.8+ for both projects
- Required packages are listed in each project's documentation

## Repository Structure

```
.
├── Task 1/                  # Risk Assessment Model
│   ├── data_processor.py
│   ├── model_trainer.py
│   ├── predictor.py
│   ├── monitor.py
│   ├── ml_ops.py
│   └── README.md
│
├── Task 2/                  # Feedback Analysis Chatbot
│   ├── rtv_chatbot.py
│   ├── app.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── llm_integration.py
│   ├── config.py
│   ├── data/
│   └── README.md
│
└── README.md                # This file
```

## Development

To contribute to either project:

1. Clone this repository
2. Set up the environment according to the project's instructions
3. Make your changes
4. Test thoroughly before submitting pull requests

## Contact

For questions or support regarding these projects, please contact the RTV technical team.
