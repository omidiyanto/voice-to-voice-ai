from flask import Flask, render_template, jsonify, request, session
from groq import Groq
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt configuration
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are JARVIS, a real-time voice assistant. Keep responses very short and conversational."
}

def get_chat_history():
    """Retrieve or initialize session-specific chat history"""
    if 'chat_history' not in session:
        session['chat_history'] = [SYSTEM_PROMPT]
    return session['chat_history'].copy()

def trim_history(history, max_length=6):
    """Maintain conversation context while preventing memory bloat"""
    return [history[0]] + history[-max_length:]

@app.route('/')
def index():
    """Initialize or renew session"""
    if 'session_id' not in session:
        session.permanent = True
        session['session_id'] = str(uuid.uuid4())
        session['chat_history'] = [SYSTEM_PROMPT]
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_handler():
    """Handle chat requests with session isolation"""
    try:
        # Get session-specific history
        history = get_chat_history()
        user_message = request.json.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Empty message", "status": "error"}), 400
        
        # Add user message to history
        history.append({"role": "user", "content": user_message})
        
        # Get AI response
        response = client.chat.completions.create(
            messages=trim_history(history),
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=100
        )
        
        # Process AI response
        assistant_response = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_response})
        
        # Update session history with trimmed context
        session['chat_history'] = trim_history(history)
        
        return jsonify({
            "response": assistant_response,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# Vercel requires this for proper initialization
if __name__ == '__main__':
    app.run()