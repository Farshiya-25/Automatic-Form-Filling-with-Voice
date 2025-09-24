# Automatic-Form-Filling-with-Voice

The Voice-Based Form Filling System is an AI-powered application that allows users to fill forms using voice commands. Instead of typing, users speak their responses, and the system automatically transcribes, extracts entities, and populates form fields in real time. The system is integrated with a lip-synced avatar for an interactive user experience.

## Features

- **Voice Input**: Users can answer form fields by speaking.

- **Speech-to-Text (STT)**: Converts audio responses to text using APIs like Groq.

- **Named Entity Recognition (NER)**: Extracts relevant entities (e.g., name, date, time, symptoms) using spaCy or LLM-based prompts.

- **Text-to-Speech (TTS)**: Generates audio questions for each field with an avatar using TTS APIs.

- **Lip-Synced Avatar**: Animated avatar lip-syncs with the generated audio for interactive UI.

- **Form Population**: Entities extracted from speech are automatically filled into the form fields.

- **Supports Multiple Fields**: Works with any number of form fields such as name, contact, date of birth, symptoms, and doctor’s name.

## Workflow

**1.Field Prompt & Avatar**: The system displays the question with the avatar image.

**2.Generate Lip-Synced Video**: Audio of the question is generated using TTS and synchronized with the avatar.

**3.Play in UI**: The lip-synced video plays in the Streamlit UI.

**4.Record User Response**: System records the user’s voice response for the current field.

**5.Transcription (STT)**: The audio response is transcribed into text using Groq API.

**6.Entity Extraction (NER)**: Extracts required entities from the transcription.

**7.Form Filling**: Extracted entities are populated in the corresponding form fields.

**8.Repeat**: Steps 1–7 repeat until all form fields are filled.

**9.Completion**: User sees a confirmation message once the form is fully populated.
## Tech Stack

**Frontend**: Streamlit (for interactive UI)

**Backend / APIs**:

- Groq API – Speech-to-Text (STT), Text-to-Speech (TTS)

- Gooey API – Lipsync video with avatar images

- spaCy – Named Entity Recognition (NER)

- Gemini LLM – Optional LLM-based entity extraction

**Python Libraries**: numpy, pandas, base64, time, dateparser

**Video Processing**: Avatar lip-sync using input images


## Installation

Clone the repository:

```bash
git clone https://github.com/Farshiya-25/Automatic-Form-Filling-with-Voice
```

Create virtual environment and activate:
```
python -m venv venv
source venv/Scripts/activate   # Windows
# or
source venv/bin/activate       # macOS/Linux
```

Install dependencies:
```
pip install -r requirements.txt
```

Set API Keys (in .env file):
```
GROQ_API_KEY=<your_groq_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
```
## Usage
1. Run the Streamlit app:
```javascript
streamlit run app.py
```
2. Click “Fill Form with Avatar”.

3. Listen to the avatar’s question.

4. Speak your response clearly.

5. The system will transcribe and fill the form automatically.

6. Continue until all fields are filled.

## Key Insights/ Feedback

**Groq STT**: Fastest transcription (~0.6s), low latency, recommended for real-time applications.

**Groq TTS**: Groq TTS provides fastest audio generation for lip-syncing avatars.

**NER**: spaCy + LLM prompts provided best reliability for entity extraction.

**User Experience**: Lip-synced avatar makes form filling engaging and interactive.

## Future Enhancements

- **Support More Form Fields**: Extend the system to handle additional types of fields beyond the current set.

- **Repeat Questions for No Response**: If the user doesn’t respond, the system will automatically repeat the question to ensure form completion.

- **Multi-Language Support**: Add support for multiple languages in STT and TTS.

- **Advanced Entity Extraction**: Integrate more LLMs for robust and context-aware NER.

- **Dynamic Avatar Customization**: Allow users to choose avatar appearance and expressions.

## Demo

[![Watch the Demo](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://drive.google.com/file/d/1my_MlWIBZ6ClArSdQ6ZKn19kBnt5OVSN/view?usp=sharing)


