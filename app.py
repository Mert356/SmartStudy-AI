import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth
import google.generativeai as genai
import fitz  
import json
import os
import requests
from dotenv import load_dotenv
import random
import plotly.express as px
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

if not firebase_admin._apps:
    cred = credentials.Certificate("edumentorai-46716-firebase-adminsdk-fbsvc-99216add5e.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

genai.configure(api_key=os.getenv("API_KEY"))

FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
if not FIREBASE_API_KEY:
    st.error("Error: FIREBASE_API_KEY is missing. Please get the Web API Key from Firebase Console and add it to the .env file.")
SIGN_IN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"

st.set_page_config(page_title="EduMentor AI", layout="wide")
st.title("EduMentor AI - Personalized Learning Platform")

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""
if "generated_questions" not in st.session_state:
    st.session_state["generated_questions"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "answered_questions" not in st.session_state:
    st.session_state["answered_questions"] = {}
if "chatbot_chain" not in st.session_state:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("API_KEY"))
    prompt_template = PromptTemplate(
        input_variables=["history", "performance_summary", "user_input"],
        template="""
        You are a friendly, motivational learning coach. The user's performance data: {performance_summary}.
        Conversation history: {history}
        User question: {user_input}
        Consider topics where the user has a success rate below 70%. Provide a concise (50-100 words), encouraging response in English. Be like a supportive friend, offering guidance and motivational tips. Explain topics clearly if needed, and build on the conversation history for context.
        """
    )
    memory = ChatMessageHistory()
    def format_history(messages):
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    st.session_state["chatbot_chain"] = RunnableSequence(
        lambda inputs: {
            "history": format_history(memory.messages),
            "performance_summary": inputs["performance_summary"],
            "user_input": inputs["user_input"]
        },
        prompt_template,
        llm
    )
    st.session_state["chatbot_memory"] = memory



def create_summary(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
        system_instruction="You are an academic assistant tasked with producing a detailed summary of the provided text. The summary should be written in a formal, scholarly tone, capturing the main objectives, technical details, and key outcomes of the EduMentor AI project for the BTK 2025 Hackathon. Focus on its educational functionalities, such as personalized learning content for students and quiz/exam generation for teachers. Limit the summary to 500 words."
    )
    
    prompt = f"Provide a detailed summary of the following text in English (max 500 words, academic tone):\n{text[:2000]}"
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text.strip()

def generate_question(pdf_summary, pdf_category, difficulty="Medium"):
    """Generates a multiple-choice question in JSON format based on the PDF summary."""
    question_types = [
        "Which of the following is the correct calculation method?",
        "What is the most appropriate analysis method for the given scenario?",
        "Which option correctly interprets the provided data?",
        "Which method is best suited for qualitative data analysis?"
    ]
    question_context = random.choice(question_types)
    
    prompt = f"""
    PDF summary: {pdf_summary}
    Topic: {pdf_category}
    Difficulty: {difficulty}
    Generate a multiple-choice question with 4 options in English, with a clear question body. 
    The question body (`q_title`) should be a concise question like '{question_context}'.
    Each option (`opt_a`, `opt_b`, `opt_c`, `opt_d`) should describe a different calculation or analysis method.
    Avoid repeating the same question; create a new, unique question.
    The `answer` field must contain the letter of the correct option (e.g., 'A', 'B', 'C', or 'D').
    The difficulty level should be specified in the `qdiff` field.
    Format:
    {{
        "q_title": "",
        "q_topic": "{pdf_category}",
        "opt_a": "",
        "opt_b": "",
        "opt_c": "",
        "opt_d": "",
        "answer": "",
        "explanation": "",
        "qdiff": "{difficulty}"
    }}
    """
    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        question_data = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        # Ensure answer is a single letter (A, B, C, or D)
        if question_data["answer"] not in ["A", "B", "C", "D"]:
            correct_option = next((k[-1].upper() for k in ["opt_a", "opt_b", "opt_c", "opt_d"] if question_data[k] == question_data["answer"]), None)
            if correct_option:
                question_data["answer"] = correct_option
            else:
                raise ValueError("Invalid answer format: must be A, B, C, or D")
        return question_data
    except Exception as e:
        st.error(f"Question generation error: {e}")
        return None
def register_user(name, email, password):
    """Registers a new user and saves their info to Firestore."""
    try:
        user = auth.create_user(email=email, password=password)
        db.collection("users").document(user.uid).set({
            "user_name": name,
            "user_email": email,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        st.success("Registration successful! Please log in.")
        return user.uid
    except Exception as e:
        st.error(f"Registration error: {e}")
        return None


def sign_in_user(email, password):
    """Signs in a user using Firebase Authentication REST API."""
    try:
        response = requests.post(SIGN_IN_URL, json={
            "email": email,
            "password": password,
            "returnSecureToken": True
        }, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_message = e.response.json().get("error", {}).get("message", str(e))
        except ValueError:
            error_message = f"Invalid server response (HTTP {e.response.status_code}): {e.response.text[:200]}... (See full response in terminal)"
            print(f"API response: {e.response.text}")
            if e.response.status_code == 404:
                error_message += " - Please ensure FIREBASE_API_KEY is correct and the web app is registered in Firebase Console."
        st.error(f"Login error: {error_message}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}. Please check your internet connection.")
        print(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return None

def upload_pdf_page():
    """Allows users to upload a PDF and save its summary."""
    st.header("Upload PDF")
    user_id = st.session_state["user_id"]
    folder_name = st.text_input("Folder Name (e.g., Statistics)")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    
    if st.button("Upload PDF"):
        if not folder_name or not pdf_file:
            st.error("Folder name and PDF are required.")
            return
        
        summary = create_summary(pdf_file)
        try:
            db.collection("user_data").document().set({
                "user_id": user_id,
                "pdf_name": pdf_file.name,
                "pdf_category": folder_name,
                "pdf_summary": summary,
                "created_at": firestore.SERVER_TIMESTAMP
            })
            st.success("PDF uploaded and summary saved!")
        except Exception as e:
            st.error(f"PDF upload error: {e}")

def generate_questions_page():
    """Allows users to select a PDF, generate questions, answer them, and view results."""
    st.header("Generate and Answer Questions")
    user_id = st.session_state["user_id"]
    
    pdfs = db.collection("user_data").where(filter=firestore.FieldFilter("user_id", "==", user_id)).stream()
    pdf_list = [p.to_dict() for p in pdfs]
    
    if not pdf_list:
        st.info("Please upload a PDF first.")
        return
    
    selected_pdf = st.selectbox("Select PDF", [p["pdf_name"] for p in pdf_list])
    difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
    quantity = st.slider("Number of Questions", 1, 20, 1)
    
    if st.button("Generate Questions"):
        pdf_data = next((p for p in pdf_list if p["pdf_name"] == selected_pdf), None)
        if pdf_data:
            questions = []
            with st.spinner("Generating questions..."):
                for i in range(quantity):
                    question = generate_question(pdf_data["pdf_summary"], pdf_data["pdf_category"], difficulty)
                    if question:
                        try:
                            doc_ref = db.collection("user_questions").document()
                            doc_ref.set({
                                "user_id": user_id,
                                "q_topic": question["q_topic"],
                                "q_title": question["q_title"],
                                "qA": question["opt_a"],
                                "qB": question["opt_b"],
                                "qC": question["opt_c"],
                                "qD": question["opt_d"],
                                "qra": question["answer"],
                                "qua": None,
                                "qex": question["explanation"],
                                "qdiff": question["qdiff"],
                                "topic_mastery": 0.0,
                                "attempt_count": 0,
                                "created_at": firestore.SERVER_TIMESTAMP
                            })
                            question["doc_id"] = doc_ref.id
                            questions.append(question)
                        except Exception as e:
                            st.error(f"Error saving question {i+1}: {e}")
                            continue
            if questions:
                st.success(f"{len(questions)} questions generated!")
                st.session_state["generated_questions"] = questions
                for q in questions:
                    st.session_state["answered_questions"][q["doc_id"]] = False
            else:
                st.error("No questions generated.")
    

    if st.session_state["generated_questions"]:
        st.subheader("Generated Questions")
        for i, q in enumerate(st.session_state["generated_questions"], 1):
            with st.expander(f"Question {i}: {q['q_title']} (Difficulty: {q.get('qdiff', 'Unknown')})"):
                st.markdown(f"**Question**: {q['q_title']}")
                st.write(f"**Topic**: {q['q_topic']}")
                st.write(f"**A)** {q['opt_a']}")
                st.write(f"**B)** {q['opt_b']}")
                st.write(f"**C)** {q['opt_c']}")
                st.write(f"**D)** {q['opt_d']}")
                
                answered = st.session_state["answered_questions"].get(q["doc_id"], False)
                if not answered:
                    user_answer = st.radio(f"Select your answer for Question {i}:", ["A", "B", "C", "D"], key=f"answer_{q['doc_id']}")
                    if st.button(f"Submit Answer for Question {i}", key=f"save_answer_{q['doc_id']}"):
                        try:
                            doc_ref = db.collection("user_questions").document(q["doc_id"])
                            doc_ref.update({
                                "qua": user_answer,
                                "attempt_count": firestore.Increment(1),
                                "topic_mastery": 1.0 if user_answer.lower() == q["answer"].lower() else 0.0
                            })
                            st.session_state["answered_questions"][q["doc_id"]] = True
                            st.success(f"Answer for Question {i} saved!")
                            st.rerun()  # Refresh to show result
                        except Exception as e:
                            st.error(f"Error saving answer: {e}")
                
                if st.session_state["answered_questions"].get(q["doc_id"], False):
                    doc_ref = db.collection("user_questions").document(q["doc_id"])
                    q_data = doc_ref.get().to_dict()
                    user_answer = q_data.get("qua", "Not answered")
                    is_correct = user_answer.lower() == q["answer"].lower()
                    st.markdown(f"**Your Answer**: {user_answer}")
                    st.markdown(f"**Correct Answer**: {q['answer']}")
                    st.markdown(f"**Result**: {'Correct! 🎉' if is_correct else 'Incorrect, try again! 😅'}")
                    st.markdown(f"**Explanation**: {q['explanation']}")

    st.subheader("Saved Questions")
    try:
        questions = db.collection("user_questions").where(filter=firestore.FieldFilter("user_id", "==", user_id)).stream()
        question_list = [q.to_dict() for q in questions]
        if not question_list:
            st.info("No saved questions yet.")
        else:
            for i, q in enumerate(question_list, 1):
                doc_id = q["doc_id"] if "doc_id" in q else db.collection("user_questions").where(filter=firestore.FieldFilter("q_title", "==", q["q_title"])).where(filter=firestore.FieldFilter("user_id", "==", user_id)).get()[0].id
                with st.expander(f"Question {i}: {q['q_title']} (Difficulty: {q.get('qdiff', q.get('difficulty', 'Unknown'))})"):
                    st.markdown(f"**Question**: {q['q_title']}")
                    st.write(f"**Topic**: {q['q_topic']}")
                    st.write(f"**A)** {q['qA']}")
                    st.write(f"**B)** {q['qB']}")
                    st.write(f"**C)** {q['qC']}")
                    st.write(f"**D)** {q['qD']}")
                    
                    answered = q.get("qua") is not None
                    if not answered:
                        user_answer = st.radio(f"Select your answer for Question {i}:", ["A", "B", "C", "D"], key=f"saved_answer_{doc_id}")
                        if st.button(f"Submit Answer for Question {i}", key=f"save_saved_answer_{doc_id}"):
                            try:
                                doc_ref = db.collection("user_questions").document(doc_id)
                                doc_ref.update({
                                    "qua": user_answer,
                                    "attempt_count": firestore.Increment(1),
                                    "topic_mastery": 1.0 if user_answer.lower() == q["qra"].lower() else 0.0
                                })
                                st.success(f"Answer for Question {i} saved!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error saving answer: {e}")
                    
                    if answered:
                        is_correct = q["qua"].lower() == q["qra"].lower()
                        st.markdown(f"**Your Answer**: {q['qua']}")
                        st.markdown(f"**Correct Answer**: {q['qra']}")
                        st.markdown(f"**Result**: {'Correct! 🎉' if is_correct else 'Incorrect, try again! 😅'}")
                        st.markdown(f"**Explanation**: {q['qex']}")
    except Exception as e:
        st.error(f"Error fetching saved questions: {e}")

def analyze_performance_page():
    """Analyzes user performance and provides chatbot guidance."""
    st.header("Performance Analysis")
    user_id = st.session_state["user_id"]
    
    try:
        questions = db.collection("user_questions").where(filter=firestore.FieldFilter("user_id", "==", user_id)).stream()
        question_list = [q.to_dict() for q in questions]
    except Exception as e:
        st.error(f"Error fetching question data: {e}")
        return
    
    if not question_list:
        st.info("No answered questions yet. Generate and answer questions from the Generate and Answer Questions page.")
        return
    
    performance_data = []
    for q in question_list:
        if q.get("qua") is not None: 
            is_correct = q["qua"].lower() == q["qra"].lower()
            performance_data.append({
                "Topic": q["q_topic"],
                "Question": q["q_title"],
                "Correct Answer": q["qra"],
                "Your Answer": q["qua"],
                "Status": "Correct" if is_correct else "Incorrect",
                "Difficulty": q.get("qdiff", q.get("difficulty", "Unknown"))
            })
    
    if not performance_data:
        st.info("No answered questions yet.")
        return
    
    df = pd.DataFrame(performance_data)
    
    topic_summary = df.groupby("Topic").agg({
        "Status": lambda x: (x == "Correct").sum() / len(x) * 100
    }).reset_index()
    topic_summary.columns = ["Topic", "Success Rate (%)"]
    
    st.subheader("Topic-Based Success Rate")
    fig = px.pie(topic_summary, values="Success Rate (%)", names="Topic", title="Topic-Based Success Distribution")
    st.plotly_chart(fig)
    
    st.subheader("Answered Questions")
    st.dataframe(df[["Topic", "Question", "Correct Answer", "Your Answer", "Status", "Difficulty"]])
    
    st.subheader("Weak Topics")
    weak_topics = topic_summary[topic_summary["Success Rate (%)"] < 70]
    if weak_topics.empty:
        st.success("You're rocking it with 70% or higher in all topics! 🚀")
    else:
        st.warning("You might want to focus on these topics:")
        st.dataframe(weak_topics)
    
    st.subheader("Chat with Your Personal Coach")
    performance_summary = json.dumps(df.groupby("Topic").agg({
        "Status": lambda x: (x == "Correct").sum() / len(x) * 100
    }).to_dict()["Status"], indent=2)
    
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("What do you want to ask your coach?"):
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        st.session_state["chatbot_memory"].add_message(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
        
        try:
            response = st.session_state["chatbot_chain"].invoke({
                "performance_summary": performance_summary,
                "user_input": user_input
            }).content
            print(f"Chatbot Response: {response}")
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            st.session_state["chatbot_memory"].add_message(AIMessage(content=response))
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            print(f"Chatbot Error: {str(e)}")
            st.error(f"Chatbot error: {e}")

if st.session_state["user_id"]:
    st.sidebar.title(f"Welcome, {st.session_state['user_name']}")
    page = st.sidebar.radio("Pages", ["Upload PDF", "Generate and Answer Questions", "Performance Analysis"])
    if page == "Upload PDF":
        upload_pdf_page()
    elif page == "Generate and Answer Questions":
        generate_questions_page()
    elif page == "Performance Analysis":
        analyze_performance_page()
else:
    st.sidebar.header("Authentication")
    auth_option = st.sidebar.radio("Option", ["Login", "Register"])
    if auth_option == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if not FIREBASE_API_KEY:
                st.error("Error: FIREBASE_API_KEY is missing. Please get the Web API Key from Firebase Console and add it to the .env file.")
            else:
                user_data = sign_in_user(email, password)
                if user_data:
                    st.session_state["user_id"] = user_data["localId"]
                    st.session_state["user_name"] = db.collection("users").document(user_data["localId"]).get().to_dict().get("user_name", "User")
                    st.success("Login successful!")
                    st.rerun()
    else:
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            register_user(name, email, password)