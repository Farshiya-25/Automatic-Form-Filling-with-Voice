import os
import io
import json
import tempfile
import urllib.request
import requests
import sounddevice as sd
import time
from groq import Groq
from dotenv import load_dotenv
from scipy.io.wavfile import write
import streamlit as st
from datetime import datetime
import base64
import re
import spacy
from dateutil import parser
from spacy.matcher import PhraseMatcher


# ---- Load ENV ----
load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")
GOOEY_API_KEY = os.getenv("GOOEY_API_KEY_1")


client = Groq(api_key=GROQ_API_KEY)

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

def listen(duration=8):
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

# ----- Load SpaCy -----
nlp = spacy.load("en_core_web_sm")

# ----- Lookups and Regex -----
GENDER_LIST = ["male", "female", "m", "f", "other", "transgender"]
SPECIALITY_LIST = [
    "cardiology","neurology","orthopedics","dermatology","pediatrics",
    "general medicine","gynecology","ent","ophthalmology","psychiatry","urology"
]
PHONE_REGEX = re.compile(r"(\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}|\d{10})")

# Common symptom phrases
SYMPTOM_LIST = [
    "fever", "cough", "cold", "sore throat", "headache", "migraine", "stomach pain",
    "abdominal pain", "nausea", "vomiting", "diarrhea", "back pain", "leg pain",
    "chest pain", "dizziness", "shortness of breath", "fatigue", "anxiety"
]

# ----- SpaCy PhraseMatcher for Symptoms -----
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(symptom) for symptom in SYMPTOM_LIST]
matcher.add("SYMPTOM", patterns)

# ===== Field-specific extractors =====
def extract_name(text):
    dr_match = re.findall(r"(?:Dr\.|Doctor)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text)
    if dr_match:
        return "Dr. " + dr_match[0]
    for ent in nlp(text).ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_dob_and_age(text):
    for ent in nlp(text).ents:
        if ent.label_ == "DATE":
            try:
                dob = parser.parse(ent.text, fuzzy=True, dayfirst=True)
                today = datetime.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                return dob.strftime("%Y-%m-%d"), str(age)
            except:
                return ent.text, None
    return None, None

def extract_gender(text):
    for g in GENDER_LIST:
        if re.search(rf"\b{g}\b", text, re.IGNORECASE):
            return g.capitalize()
    return None

def extract_phone(text):
    match = PHONE_REGEX.search(text)
    if match:
        return match.group()
    return None

def extract_symptoms(text):
    doc = nlp(text)
    matches = matcher(doc)
    symptoms = [doc[start:end].text for match_id, start, end in matches]
    if not symptoms:
        return None
    
    # If you only want the first symptom as a plain string
    #return symptoms[0]

    # OR if you want all symptoms in a single string (comma-separated)
    return ", ".join(symptoms)

def extract_speciality(text):
    for s in SPECIALITY_LIST:
        if re.search(rf"\b{s}\b", text, re.IGNORECASE):
            return s.title()
    return None


def extract_appointment(text):
    date_part, time_part = None, None
    doc = nlp(text)

    # Try with spaCy entities first
    for ent in doc.ents:
        if ent.label_ == "DATE":
            date_part = ent.text.strip()
        elif ent.label_ == "TIME":
            time_part = ent.text.strip()

    try:
        if date_part or time_part:
            # Combine if both found
            if date_part and time_part:
                dt = parser.parse(f"{date_part} {time_part}", fuzzy=True, dayfirst=True)
            elif date_part:
                dt = parser.parse(date_part, fuzzy=True, dayfirst=True)
            elif time_part:
                dt = parser.parse(time_part, fuzzy=True, dayfirst=True)
            return dt.strftime("%Y-%m-%d %H:%M") if time_part else dt.strftime("%Y-%m-%d")
        else:
            # Fallback: let parser handle the whole text
            dt = parser.parse(text, fuzzy=True, dayfirst=True)
            return dt.strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        return text  # fallback return original text if parsing fails


def extract_entity(field, text):
    if field == "Patient Name":
        return extract_name(text)
    elif field == "Age/Date of Birth":
        dob, age = extract_dob_and_age(text)
        return {"DOB": dob, "AGE": age}
    elif field == "Gender":
        return extract_gender(text)
    elif field == "Contact Number":
        return extract_phone(text)
    elif field == "Speciality":
        return extract_speciality(text)
    elif field == "Doctor Name":
        return extract_name(text)
    elif field == "Date and Time":
        return extract_appointment(text)
    elif field == "Reason for Visit / Symptoms":
        return extract_symptoms(text)
    return None

#-------------- FIELD PROMPTS-------------

fields = ["Patient Name",
          "Date of Birth",
          "Age",
          "Gender",
          "Contact Number",
          "Reason for Visit / Symptoms",
          "Speciality",
          "Doctor Name",
          "Date and Time"]

FIELD_PROMPTS = {
    "Patient Name": "Please tell me your name?",
    "Date of Birth": "Can you tell me your date of birth?",
    "Gender": "What is your gender?",
    "Contact Number": "Please share your contact number.",
    "Reason for Visit / Symptoms": "Please describe your symptoms.",
    "Speciality": "Which speciality would you like to consult?",
    "Doctor Name": "Do you have a preferred doctors name?",
    "Date and Time": "When would you like to book the appointment?"
}


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
overall_start = time.time()

if st.button("🎙️ Fill Form with Voice"):
    st.session_state.filling = True
    st.session_state.current_field = 0
    st.session_state.start_time = time.time()


# ---- Voice Form Filling ----
if st.session_state.filling and st.session_state.current_field < len(fields):
    field_name = fields[st.session_state.current_field]    
    question = FIELD_PROMPTS.get(field_name)
    
    # Avatar asks question
    if question:
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

        temp_filename = listen(duration=5)

        field_start = time.time()
        answer = transcribe(temp_filename)
        field_end = time.time()

        print(f"⏱️ Transcription time for {field_name}: {field_end - field_start:.2f} sec")

        # Extract value
        if field_name=="Date of Birth":
            # Ask DOB first and fill both DOB and AGE
            if field_name == "Date of Birth":
                dob, age = extract_dob_and_age(answer)
                st.session_state.form["Date of Birth"] = dob
                st.session_state.form["Age"] = age
                print(f"📌 Date of Birth: {dob}, Age: {age}")

        elif field_name == "Reason for Visit / Symptoms":
            symptoms = extract_symptoms(answer)
            st.session_state.form["Reason for Visit / Symptoms"] = symptoms
            print(f"📌 Symptoms: {symptoms}")

        else:
            value = extract_entity(field_name, answer)
            st.session_state.form[field_name] = value
            print(f"📌 {field_name}: {value}")

    else:
        # No prompt, skip this field (Age is already auto-filled)
        pass

    st.session_state.current_field += 1
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
