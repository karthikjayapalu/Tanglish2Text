import streamlit as st
import json
import re
import tempfile
import os
import requests
from groq import Groq
from typing import Dict, Union
from elevenlabs.client import ElevenLabs

# Streamlit UI Configuration
st.set_page_config(
    page_title="Tamil-English Voice Processor", 
    layout="wide",
    page_icon="🎤"
)

# Configuration
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
except (st.errors.StreamlitSecretNotFoundError, KeyError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    if not GROQ_API_KEY:
        st.error("GROQ API Key not found")
        st.stop()

# Initialize clients
client = Groq(api_key=GROQ_API_KEY)
if ELEVENLABS_API_KEY:
    eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def transcribe_with_groq(audio_file) -> Union[str, None]:
    """Transcribe audio using Groq's Whisper model"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        return transcription
    except Exception as e:
        st.error(f"Groq transcription failed: {str(e)}")
        return None
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def transcribe_with_elevenlabs(audio_file) -> Union[str, None]:
    """Transcribe audio using ElevenLabs API"""
    if not ELEVENLABS_API_KEY:
        st.error("ElevenLabs API key not configured")
        return None

    tmp_path = None  # Initialize to avoid reference error
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        # Prepare the API request
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY
        }

        # Use context manager to ensure the file is closed before deletion
        with open(tmp_path, 'rb') as f:
            files = {
                'file': f
            }
            data = {
                # 'model_id': 'scribe_v1'
                'model_id':'scribe_v1_experimental'
            }

            # Make the API request
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()

        # Parse the response
        result = response.json()
        return result.get('text', '')

    except Exception as e:
        st.error(f"ElevenLabs transcription failed: {str(e)}")
        if 'response' in locals():
            st.error(f"API Response: {response.text}")
        return None

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as cleanup_error:
                st.warning(f"Could not delete temporary file: {cleanup_error}")


def extract_entities_with_llama(text: str) -> Dict:
    """Entity extraction using few-shot structured prompting"""
    system_prompt = """
    You are an expert at extracting structured information from Tamil-English (Tanglish) text.
    Always return a JSON object with these exact fields:
    - person_name: array of Tamil/English names
    - place: array of locations
    - phone_number: array of numbers in +91XXXXXXXXXX format
    - skills: array of skills in Tamil/English
    - intent: one of [Introduction, JobQuery, Complaint, InformationRequest, Other]
    - intent_confidence: number between 0-1

    Examples:

    Example 1:
    Text: "என் பெயர் ராஜா, நான் திரிச்சியில் வசிக்கிறேன். என் எண் +917777777777. நான் விவசாயி."
    Output: {
        "person_name": ["ராஜா"],
        "place": ["திரிச்சி"],
        "phone_number": ["+917777777777"],
        "skills": ["விவசாயி"],
        "intent": "Introduction",
        "intent_confidence": 0.9
    }

    Example 2:
    Text: "I'm Kumar from Chennai. My skills are Python and Machine Learning. Call me at 9876543210."
    Output: {
        "person_name": ["Kumar"],
        "place": ["Chennai"],
        "phone_number": ["+919876543210"],
        "skills": ["Python", "Machine Learning"],
        "intent": "JobQuery",
        "intent_confidence": 0.85
    }

    Example 3:
    Text: "The roads in Madurai are very bad. Please fix them."
    Output: {
        "person_name": [],
        "place": ["Madurai"],
        "phone_number": [],
        "skills": [],
        "intent": "Complaint",
        "intent_confidence": 0.95
    }

    Now analyze this text:
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            model="llama3-70b-8192",
            response_format={"type": "json_object"},
            temperature=0.1
        )

        # Extract JSON from response
        raw_response = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        
        if not json_match:
            raise ValueError("No JSON found in response")
        
        entities = json.loads(json_match.group())
        
        # Validate and clean the response
        return {
            "person_name": entities.get("person_name", []),
            "place": entities.get("place", []),
            "phone_number": [re.sub(r'[^\d+]', '', num) for num in entities.get("phone_number", [])],
            "skills": entities.get("skills", []),
            "intent": entities.get("intent", "Other"),
            "intent_confidence": min(max(float(entities.get("intent_confidence", 0)), 0), 1)
        }
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            "person_name": [],
            "place": [],
            "phone_number": [],
            "skills": [],
            "intent": "unknown",
            "intent_confidence": 0
        }

# UI Elements
st.title("🎤 Tamil+English Voice to Structured Data")

with st.expander("ℹ️ How to use"):
    st.write("""
    1. Upload an audio file (MP3/WAV format)
    2. Select transcription model
    3. Click 'Process Audio' button
    4. View transcription and extracted information
    """)

# Model selection
transcription_model = st.radio(
    "Select Transcription Model",
    ["Groq (Whisper)", "ElevenLabs"],
    index=0,
    help="Choose which model to use for transcription"
)

audio_file = st.file_uploader(
    "Upload audio file", 
    type=["mp3", "wav"],
    help="Supported formats: MP3, WAV (max 25MB)"
)

if audio_file and st.button("Process Audio", type="primary"):
    with st.spinner("Transcribing..."):
        if transcription_model == "Groq (Whisper)":
            transcription = transcribe_with_groq(audio_file)
        else:
            transcription = transcribe_with_elevenlabs(audio_file)
    
    if transcription:
        st.subheader("Transcription")
        st.text_area("Transcript", transcription, height=150, label_visibility="collapsed")
        
        with st.spinner("Analyzing content..."):
            entities = extract_entities_with_llama(transcription)
        
        st.subheader("Structured Output")
        
        col1, col2 = st.columns(2)
        with col1:
            st.json(entities, expanded=True)
        
        with col2:
            st.metric("Detected Intent", 
                     value=entities["intent"],
                     help=f"Confidence: {entities['intent_confidence']:.2f}")
            
            for category, emoji in [("person_name", "👤"), 
                                  ("place", "📍"), 
                                  ("phone_number", "📞"), 
                                  ("skills", "🛠️")]:
                if entities[category]:
                    st.write(f"{emoji} **{category.replace('_', ' ').title()}**")
                    for item in entities[category]:
                        st.write(f"- {item}")