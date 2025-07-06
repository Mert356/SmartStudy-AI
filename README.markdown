# SmartStudy-AI

## Overview
SmartStudy-AI is an AI-powered educational platform that delivers personalized learning content, generates quizzes from PDF summaries, and tracks performance with interactive charts. Built with Streamlit, it uses Google Gemini 2.5 Flash for AI capabilities and Firebase for authentication and data storage.

## Features
- Personalized study materials based on student proficiency
- Multiple-choice quiz generation with adjustable difficulty
- Performance analysis with Plotly charts
- Motivational chatbot coach powered by LangChain
- PDF content extraction and summarization
- Secure user authentication

## Prerequisites
- Python 3.9 or higher
- Firebase project with Authentication and Firestore
- Google Cloud account with Gemini API access

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mert356/SmartStudy-AI.git
   cd SmartStudy-AI
   ```

2. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```plaintext
   API_KEY=your-google-gemini-api-key
   FIREBASE_API_KEY=your-firebase-web-api-key
   ```
   - Get `API_KEY` from Google Cloud Console.
   - Get `FIREBASE_API_KEY` from Firebase Console (Project Settings > General > Web API Key).
   - Place Firebase Admin SDK JSON file in the project root.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   Access at `http://localhost:8501`.

## Usage
1. **Register/Login**: Use the sidebar to sign up or log in.
2. **Upload PDF**: Go to "Upload PDF", add a folder name, and upload a PDF to generate a summary.
3. **Generate Questions**: Select a PDF, choose difficulty, and answer questions with instant feedback.
4. **Analyze Performance**: View success rates and get personalized advice from the chatbot.

## Project Structure