# EmpathyBot API  

A Flask-based API that provides emotional support by detecting user emotions and generating empathetic, supportive responses using machine learning models from Hugging Face.

## Features  
- **Emotion Detection**: Analyzes input text to identify emotions like sadness, anger, or fear.  
- **Supportive Responses**: Generates tailored replies based on detected emotions using a pre-trained conversational model.  
- **Easy Integration**: Simple REST API endpoint for interacting with the bot.  

## Technologies Used  
- **Flask**: Web framework for building the API.  
- **Hugging Face Transformers**: Used `DialoGPT-small` for conversation and `DistilRoBERTa` for emotion detection.  
- **Python**: Backend logic and API implementation.  

## Installation  

 Clone the repository:  
   ```bash
   git clone https://github.com/mrayushmehrotra/empathy-bot.git
   cd empathy-bot
  ```

2. Install Dependencies and run the Server
```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   python app.py
```
