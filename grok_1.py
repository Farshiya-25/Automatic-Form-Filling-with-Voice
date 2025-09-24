import os
import io
import json
import tempfile
import urllib.request
import requests
import sounddevice as sd
import time
from groq import Groq
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from scipy.io.wavfile import write
import google.generativeai as genai
import streamlit as st
import google.generativeai as genai
from datetime import datetime
import dateparser
import base64


# ---- Load ENV ----
load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")
GEMINI_API_KEY = os.getenv("Gemini_API_KEY_3")
GOOEY_API_KEY = os.getenv("GOOEY_API_KEY_1")


client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

face_image_path =r"C:\Users\Abdul\OneDrive\Desktop\Speech_Text\avatar\cropped_half_body.jpg"

# ----------------- GOOEY LIPSYNC -----------------
def lipsync_with_avatar(text, face_image_path="face.jpg"):
    """
    Takes a text question, generates TTS audio, 
    sends it with face image to Gooey API for lipsync video.
    Returns path to downloaded video.
    """
    try:
        # 1. Generate TTS with Groq
        response = client.audio.speech.create(
            model="playai-tts",
            voice="Arista-PlayAI",
            input=text,
            response_format="wav"
        )
        audio_data = io.BytesIO(response.read())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data.read())
            audio_path = temp_audio.name

        # 2. Call Gooey API
        url = "https://api.gooey.ai/v2/Lipsync/form/"
        with open(face_image_path, "rb") as img_file, open(audio_path, "rb") as audio_file:
            files = [("input_face", img_file), ("input_audio", audio_file)]
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {GOOEY_API_KEY}"},
                files=files,
                data={"json": json.dumps({})},
            )

        if r.ok:
            result = r.json()
            video_url = result.get("output", {}).get("output_video")
            if video_url:
                output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                urllib.request.urlretrieve(video_url, output_file)
                return output_file
        else:
            print("❌ Gooey API error:", r.text)
            return None
    except Exception as e:
        print("❌ Lipsync error:", e)
        return None

# ----------------- RECORDING -----------------

def listen(duration=5):
    samplerate = 16000
    channels = 1

    st.write("🎤 Speak now...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="int16")
    sd.wait()

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_filename = temp_file.name
    write(temp_filename, samplerate, recording)

    return temp_filename


# ----------------- TRANSCRIBE -----------------

def transcribe(temp_filename):
    try:
        # Transcribe with Groq Whisper
        with open(temp_filename, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                language="en"
            )

        text = transcription.text.strip()
        print(f"✅ You said: {text}")
        return text
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return ""

# ----------------- ENTITY EXTRACTION -----------------

def extract_entities(field_name,field_text):
    try:
        prompt = f"""Please extract a proper {field_name} from this text: {field_text}. 
        Only return the value.
        Do not include any explanation or extra text."""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return field_text

def parse_date(date_text):
    """
    Convert a natural language date string into a Python datetime object.
    """
    dt = dateparser.parse(date_text)
    return dt

def parse_dob(dob_text):
    dt = dateparser.parse(dob_text)
    if not dt:
        try:
            dt = datetime.strptime(dob_text, "%B %d, %Y")  # e.g. January 21, 1998
        except:
            return None
    return dt.date()

# ----------------- FIELD PROMPTS -----------------

FIELD_PROMPTS = {
    "Patient Name": "Please tell me your name?",
    "Age/Date of Birth": "Can you tell me your date of birth or your age?",
    "Gender": "What is your gender?",
    "Contact Number": "Please share your contact number.",
    "Reason for Visit / Symptoms": "Please describe your symptoms.",
    "Speciality": "Which speciality would you like to consult?",
    "Doctor Name": "Do you have a preferred doctors name?",
    "Date and Time": "When would you like to book the appointment?"
}

fields = list(FIELD_PROMPTS.keys())


# ------------------STREAMLIT------------------

st.set_page_config(page_title="Voice-based Form", page_icon="📝", layout="centered")
st.title("📝 Voice-based Registration Form")


if "form" not in st.session_state:
    st.session_state.form = {field: "" for field in fields}

if "current_field" not in st.session_state:
    st.session_state.current_field = 0

if "filling" not in st.session_state:
    st.session_state.filling = False

# --- Avatar placeholder at the top ---
avatar_placeholder = st.empty()

# Always show avatar image before filling starts
# Initially show static avatar image
# if "started" not in st.session_state:
#     st.session_state.started = False
#     avatar_placeholder.image(face_image_path, caption="Your Virtual Assistant", width=300)

def show_avatar():
    avatar_placeholder.image(face_image_path, caption="Your Virtual Assistant", width=300)

show_avatar()

# Show form textboxes (always visible)
cols = st.columns(2)
for i, field in enumerate(fields):
    col = cols[i % 2]  
    with col:
        st.text_input(field, value=st.session_state.form[field], key=field)

# Button to start filling

if st.button("🎙️ Fill Form with Avatar"):
    st.session_state.filling = True
    st.session_state.current_field = 0
    st.session_state.start_time = time.time()


# ---- Voice Form Filling ----
if st.session_state.filling and st.session_state.current_field < len(fields):
    field_name = fields[st.session_state.current_field]    
    question = FIELD_PROMPTS.get(field_name)

    # Avatar asks question
    video_path = lipsync_with_avatar(question, face_image_path=face_image_path)
    if video_path:
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        video_html = f"""
        <video width="300" autoplay>
          <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        """
        avatar_placeholder.markdown(video_html, unsafe_allow_html=True)

    time.sleep(3)
    show_avatar()    

    temp_filename = listen(duration=6)

    field_start = time.time()
    answer = transcribe(temp_filename)
    field_end = time.time()

    print(f"⏱️ Transcription time for {field_name}: {field_end - field_start:.2f} sec")

    entities = extract_entities(field_name,answer)

    if "birth" in field_name.lower() or "date and time" in field_name.lower():
        dt = parse_date(entities)
        if dt:
            if "birth" in field_name.lower():
                value = dt.date()    # keep only date
            else:
                value = dt          # full datetime
        else:
            value = entities
    else:
        value = entities

    print(f"📌 {field_name.capitalize()}: {value}")
    st.session_state.form[field_name] = value

    st.session_state.current_field += 1
    # Rerun to update UI
    st.rerun()

# When finished
if st.session_state.current_field == len(fields):
    st.success("✅ Form filled successfully with voice input!")
    st.session_state.filling = False
    
    show_avatar()
    if st.session_state.start_time:
        total_time = time.time() - st.session_state.start_time
        print(f"⏱️ Total runtime for form filling: {total_time:.2f} sec")
        st.session_state.start_time = None 
