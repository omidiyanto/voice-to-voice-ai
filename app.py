from flask import Flask, render_template, jsonify, request, session
from groq import Groq
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt template
SYSTEM_PROMPT = "You are JARVIS, a friendly and chill super intelligence AI created by O.Midiyanto. Keep responses very short and conversational."

def get_chat_history():
    """Initialize or retrieve session-specific chat history"""
    if 'chat_history' not in session:
        session['chat_history'] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return session['chat_history']

def trim_history(history, max_length=6):
    """Keep only the system prompt + last max_length messages"""
    return [history[0]] + history[-max_length:]

@app.route('/')
def index():
    # Initialize session with unique ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        history = get_chat_history()
        user_message = request.json['message']
        
        # Add user message to history
        history.append({"role": "user", "content": user_message})
        
        # Get AI response
        response = client.chat.completions.create(
            messages=trim_history(history),
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=100
        )
        
        # Add AI response to history
        assistant_response = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_response})
        
        # Update session storage and trim history
        session['chat_history'] = trim_history(history)
        
        return jsonify({
            "response": assistant_response,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
