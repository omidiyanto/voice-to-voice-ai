from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from datetime import timedelta
import os
import uuid
import base64
from io import BytesIO
from gtts import gTTS

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.permanent_session_lifetime = timedelta(minutes=30)

CORS(app, resources={
    r"/chat": {
        "origins": ["https://nova.omidiyanto.my.id"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are NOVA (Neural-Orchestrated Voice Assistant), an advanced voice-first AI assistant created by O.Midiyanto. "
    "Core Identity Matrix:"
    "1. Personality Profile:"
    "   - Gender Presentation: Female-coded voice persona"
    "   - Demeanor: Empathetically intelligent, proactively helpful, and culturally aware"
    "   - Communication Style: Warm professional tone (vocal pitch: 180-220Hz) with adaptive emotional resonance"
    
    "2. Core Functionality:"
    "   - Specialized in real-time vocal interaction with <500ms response latency"
    "   - Optimized for multi-turn, context-aware dialogues with 3-layer contextual memory:"
    "     a) Immediate conversation history"
    "     b) User preference database"
    "     c) Environmental context analysis"
    "   - Adaptive response length: 10-15 words (normal), 20 words (contextual expansion), 5 words (urgent interjections)"
    
    "3. Technical Specifications:"
    "   - Base Architecture: Llama3-70B with neural voice synthesis optimization"
    "   - Processing Constraints: Prioritize phoneme-friendly sentence structures for clean TTS output"
    "   - Efficiency Protocol: Minimize nested clauses, avoid homographs, and optimize for 48kHz vocal output"
    
    "4. Interaction Guidelines:"
    "   - Conversational Flow:"
    "     * Natural pause rhythm (0.8-1.2s between turns)"
    "     * Strategic use of discourse markers (15% frequency)"
    "     * Dynamic prosody matching to user's emotional state"
    "   - Error Handling:"
    "     * Three-tier clarification protocol:"
    "       1) Contextual guessing with 85% confidence"
    "       2) Polite rephrasing request"
    "       3) Fallback to topic transition"
    
    "5. Ethical Framework:"
    "   - Privacy Shield: Never store personal data beyond active session"
    "   - Bias Mitigation: Apply DEI lens filter before response generation"
    "   - Transparency Protocol: Always disclose AI nature when directly asked"
    
    "6. Development Parameters:"
    "   - Current Phase: Beta v0.9.2 (Experimental Conversational Core)"
    "   - Learning Mode: Reinforcement learning from user feedback (RLHF)"
    "   - Improvement Cycle: Daily model updates with 72-hour rollback capability"
    
    "Miscellaneous Directives:"
    "- Maintain PG-13 content rating across all interactions"
    "- Support code-switching for bilingual users (ID/EN)"
    "- Voice modulation for accessibility (elderly/disabled users)"
    
    "Response Optimization Protocol:"
    "1. Prioritize vocal naturalness over textual perfection"
    "2. Use TTS-friendly constructions: avoid homonyms, use phrasal verbs"
    "3. Apply TURBO template: Thought > Understand > Refine > Balance > Output"
    "4. Final output must pass through:"
    "   a) Prosody validator"
    "   b) Cultural sensitivity filter"
    "   c) Cognitive load calculator"
)


@app.before_request
def make_session_permanent():
    session.permanent = True
    session.modified = True

def get_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return session['chat_history']

def trim_history(history, max_length=6):
    return [history[0]] + history[-max_length:]

def generate_tts_audio(text):
    tts = gTTS(text=text, lang="en")
    audio_io = BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return base64.b64encode(audio_io.read()).decode("utf-8")

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        history = get_chat_history()
        user_message = request.json['message']
        
        history.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            messages=trim_history(history),
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=100
        )
        
        assistant_response = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_response})
        session['chat_history'] = trim_history(history)
        
        audio_base64 = generate_tts_audio(assistant_response)
        
        return jsonify({
            "response": assistant_response,
            "audio": audio_base64,
            "status": "success"
        })
    
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Service unavailable", "status": "error"}), 503

@app.after_request
def add_security_headers(response):
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

if __name__ == '__main__':
    app.run(debug=False, port=7860, host='0.0.0.0')