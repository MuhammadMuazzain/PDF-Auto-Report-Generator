import os
import uuid
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Union
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (for local development)
load_dotenv()

app = FastAPI()

# CORS setup for production and development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://*.vercel.app", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients with environment variables
try:
    # Shorter timeout to fail faster when overloaded
    http_client = httpx.Client(timeout=20.0, follow_redirects=True)
    anthropic_client = Anthropic(
        api_key=os.getenv("CLAUDE_API_KEY"),
        http_client=http_client
    )
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    INDEX_NAME = "regulatory-cases"
except Exception as e:
    logger.error(f"Failed to initialize clients: {e}")
    raise RuntimeError(f"Failed to initialize clients: {e}")

# Initialize embedding model at startup
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {e}")
    raise RuntimeError(f"Failed to initialize embedding model: {e}")

# Verify Pinecone index
try:
    index = pinecone_client.Index(INDEX_NAME)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone index: {e}")
    raise RuntimeError(f"Failed to initialize Pinecone index: {e}")

# Store conversation sessions (note: in-memory, not persistent in serverless)
conversation_sessions = {}

class ChatRequest(BaseModel):
    message: str
    conversation: List[dict]
    sessionId: Optional[str] = None

class ConversationMessage(BaseModel):
    role: str
    content: str

def get_embedding(text):
    try:
        return embedding_model.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

# OPTIMIZED & CLEAN HELPER FUNCTIONS

def is_positive_response(response):
    """Simple check for positive responses"""
    response = response.lower().strip()
    return any(word in response for word in ['yes', 'yeah', 'yep', 'ok', 'sure', 'ready', 'go', 'start'])

def get_conversation_state(conversation):
    """Simple conversation state detection"""
    user_messages = [msg['content'] for msg in conversation if msg['role'] == 'user']
    
    if len(user_messages) == 0:
        return "initial"
    elif len(user_messages) == 1:
        return "questions_start" if is_positive_response(user_messages[0]) else "not_ready"
    elif len(user_messages) == 2:
        # Handle the case where user said "no" first, then came back with "yes"
        if not is_positive_response(user_messages[0]) and is_positive_response(user_messages[1]):
            return "questions_start"
        else:
            return "questions_continue"
    else:
        return "questions_continue"

def generate_follow_up_questions(conversation):
    """Simple question generator - much cleaner logic"""
    
    # Define the structured questions in order
    structured_questions = [
        "I'd like to start by getting to know you a bit better. Could you tell me about your work? I'm interested in your profession, how long you've been qualified, your typical working hours, and what your working pattern is like.",
        "I need to understand who will be receiving this mitigation statement. Could you let me know who you intend to present this to?",
        "If you're unable to attend the hearing in person, would you mind sharing the reasons why? This can be important context for the panel or court.",
        "Now, let's talk about the situation you're facing. What specific charges or allegations are you dealing with, and who has brought them forward - is it your employer, a regulatory tribunal, or a court?",
        "I know this might be difficult to discuss, but could you walk me through what happened? Take your time - I'm here to listen and help.",
        "Help me understand what led to this situation. What were you thinking at the time? Was this something intentional, perhaps a lapse in judgment, or maybe due to lack of training or oversight? There's no judgment here - I just need to understand.",
        "When the issue first came to light, were you able to admit to the offence or charge straight away?",
        "How did you handle the situation when it was discovered? Were you able to cooperate with your employer, the regulator, or the police?",
        "Do you feel able to acknowledge your role and responsibility in what happened? This is often an important part of the process.",
        "Reflecting on this experience, what lessons have you learned? Can you share any insights about how your actions may have affected others - perhaps clients, colleagues, or the public?",
        "I'd like to understand the personal impact on you. How do you think these allegations or charges will affect you moving forward?",
        "Let's talk about your personal circumstances, as these can be relevant for mitigation. Are you married, single, or in a relationship?",
        "Do you have any children? Family circumstances can be important context.",
        "If you do have children, do any of them have specific needs such as Autism or ADHD? This kind of information can be relevant.",
        "I hope you don't mind me asking - do you have any health conditions that might be relevant to your situation?",
        "Are you the sole earner in your household? This can be important when considering the impact of any sanctions.",
        "Are you currently receiving any social security benefits or disability benefits such as income support?",
        "Do you have any debts or financial obligations that might be relevant?",
        "If you do have debts, do you have a payment plan in place to manage them?",
        "Were there any personal circumstances that might have contributed to the situation? I'm thinking of things like physical or mental health issues, burnout, or work-related pressure.",
        "Sometimes workplace factors can contribute to these situations. Were there any systemic or organizational issues involved - perhaps understaffing, lack of training, unclear protocols, feeling unsupported, or pressure from management or peers?",
        "Looking ahead, how would a disciplinary sanction such as suspension or conditions on your practice affect your livelihood?",
        "This is important for your statement - are you able to express genuine remorse for what has happened?",
        "Have you undertaken any reflective work or participated in reflective practice since this occurred? This can be valuable.",
        "If you have completed any reflective work, would you be comfortable sharing it with the panel or court as an appendix to your mitigation statement?",
        "Have you undertaken any courses, continuing professional development, or remedial training since the allegations arose? If so, what were they focused on, and do you have proof of attendance?",
        "Can you tell me about any past involvement you've had in teaching, mentoring, or quality improvement initiatives? This helps show your commitment to the profession.",
        "Have you made any changes to your practice or decision-making processes as a result of this experience?",
        "Prior to this incident, did you have an unblemished professional record?",
        "How have you contributed to your profession or community over the years? This can be important context for the panel.",
        "We touched on this earlier, but how do you think these allegations or charges will impact you personally and professionally?",
        "Are you able to obtain good character references from colleagues or clients to present to the panel or court? Please note that any character referee must state in their reference that they are aware of the allegations.",
        "Finally, how can you reassure the panel or court that this won't happen again? What steps have you taken or will you take?",
        "Is there anything else you'd like to share or add that you think would be important for the panel or court to know about your situation?"
    ]
    
    state = get_conversation_state(conversation)
    
    # Handle different states
    if state == "initial":
        return ["readiness_check"]
    elif state == "not_ready":
        return ["not_ready"]
    elif state == "questions_start":
        return [structured_questions[0]]
    elif state == "questions_continue":
        # Get user messages and skip readiness response(s)
        user_messages = [msg['content'] for msg in conversation if msg['role'] == 'user']
        
        # Determine how many messages to skip based on the conversation pattern
        skip_count = 1  # Default: skip just the initial "yes"
        
        # If user said "no" first, then "yes", skip both
        if (len(user_messages) >= 2 and 
            not is_positive_response(user_messages[0]) and 
            is_positive_response(user_messages[1])):
            skip_count = 2
        
        actual_responses = user_messages[skip_count:] if len(user_messages) > skip_count else []
        
        # Check if last answer is too short
        if actual_responses and len(actual_responses[-1].strip()) < 2:
            return ["Could you provide a bit more detail? Even a sentence or two would be helpful."]
        
        # Count good answers
        good_answers = 0
        for response in actual_responses:
            if not len(response.strip()) < 2:
                good_answers += 1
        
        # Return next question or finish
        if good_answers < len(structured_questions):
            return [structured_questions[good_answers]]
        else:
            return []  # All questions answered
    
    return []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        session_id = request.sessionId or str(uuid.uuid4())
        
        # Clean up conversation
        cleaned_conversation = [
            msg for msg in request.conversation 
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg and msg['content'].strip()
        ]
        
        conversation_sessions[session_id] = cleaned_conversation
        
        # Simple logic flow
        state = get_conversation_state(cleaned_conversation)
        questions = generate_follow_up_questions(cleaned_conversation)
        
        # If no more questions, generate the statement
        if not questions and state == "questions_continue":
            # Generate mitigation statement...
            pass
        else:
            # Handle different states
            if state == "initial":
                response_text = "Hi, welcome to your consultation. This should take about 15 minutes to complete as I need important information. Are you ready to start?"
            elif state == "not_ready":
                response_text = "No problem. Come back when you are ready."
            elif state == "questions_start":
                response_text = "Awesome, let's go. " + questions[0] if questions else "Let's get started!"
            elif state == "questions_continue":
                response_text = "Thank you for sharing that with me. " + questions[0] if questions else "Thank you for all that information."
            else:
                response_text = "Thank you for that information. Could you tell me more about your situation?"
            
            return {
                "response": response_text,
                "sessionId": session_id,
                "isFinal": False
            }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)