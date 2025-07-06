EduMentor AI: Personalized Learning Platform
Overview
EduMentor AI is an AI-powered educational platform designed to enhance learning experiences for students and support educators. It leverages the Google Gemini 2.5 Flash model to deliver personalized learning content, generate quizzes, and provide performance analytics. Built with Streamlit for an intuitive user interface and integrated with Firebase for secure user authentication and data storage, EduMentor AI offers a seamless way to engage with educational materials.
Features

Personalized Learning Content: Generates tailored study materials based on individual student proficiency levels using the Gemini API.
Quiz and Exam Generation: Creates multiple-choice questions from uploaded PDF summaries, with customizable difficulty levels (Easy, Medium, Hard).
Performance Analysis: Tracks user performance with topic-based success rates, visualized through interactive Plotly charts.
Chatbot Coach: Provides motivational guidance and topic-specific advice based on user performance, powered by LangChain and Gemini.
Secure Authentication: Manages user registration and login via Firebase Authentication.
PDF Processing: Extracts and summarizes content from uploaded PDFs using PyMuPDF for efficient learning material management.

Prerequisites
To run EduMentor AI on your local machine, ensure the following are installed:

Python 3.9 or higher
A Firebase project with Authentication and Firestore enabled
A Google Cloud account with access to the Gemini API

Installation
Follow these steps to set up and run EduMentor AI locally:

Clone the Repository:
git clone https://github.com/your-username/edumentor-ai.git
cd edumentor-ai


Install Dependencies:Create a virtual environment and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the project root and add the following:
API_KEY=your-google-gemini-api-key
FIREBASE_API_KEY=your-firebase-web-api-key


Obtain the API_KEY from the Google Cloud Console.
Obtain the FIREBASE_API_KEY from the Firebase Console (Project Settings > General > Web API Key).
Place your Firebase Admin SDK JSON file (e.g., edumentorai-46716-firebase-adminsdk-fbsvc-99216add5e.json) in the project root.


Run the Application:Start the Streamlit app:
streamlit run app.py

Access the app at http://localhost:8501 in your web browser.


Usage

Register or Log In:
Use the sidebar to create an account or log in with your email and password.


Upload a PDF:
Navigate to the "Upload PDF" page.
Enter a folder name (e.g., "Statistics") and upload an educational PDF.
The system will generate a summary of the PDF content.


Generate and Answer Questions:
Go to the "Generate and Answer Questions" page.
Select a PDF, choose a difficulty level, and specify the number of questions.
Answer the generated multiple-choice questions and view immediate feedback with explanations.


Analyze Performance:
Visit the "Performance Analysis" page to see your topic-based success rates in interactive charts.
Use the chatbot coach to ask questions and receive personalized learning advice based on your performance.



Project Structure

app.py: Main application file containing the Streamlit interface and core logic.
requirements.txt: List of Python dependencies.
.env: Environment variables for API keys (not tracked in Git).
edumentorai-46716-firebase-adminsdk-fbsvc-99216add5e.json: Firebase Admin SDK credentials (not tracked in Git).

Notes

Ensure the Firebase Admin SDK JSON file and .env file are not included in version control. Add them to your .gitignore:*.json
.env
venv/
__pycache__/


The application requires an active internet connection to interact with Firebase and the Gemini API.

Contributing
Contributions are welcome! Fork the repository, create a new branch, and submit a pull request with your changes. Ensure code adheres to PEP 8 guidelines.
License
This project is licensed under the MIT License.
Contact
For inquiries, please contact your-email@example.com.