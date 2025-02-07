import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os

def load_model():
    model = BertForSequenceClassification.from_pretrained("bert_spam_model")
    tokenizer = BertTokenizer.from_pretrained("bert_spam_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio_format = audio_file.name.split(".")[-1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name
    
    # Convert to WAV if not already in WAV format
    if audio_format != "wav":
        sound = AudioSegment.from_file(temp_audio_path)
        temp_audio_path = temp_audio_path.replace(audio_format, "wav")
        sound.export(temp_audio_path, format="wav")
    
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Could not understand the audio."
        except sr.RequestError:
            text = "Speech-to-text service unavailable."
    
    os.remove(temp_audio_path)  # Clean up
    return text

def predict_spam(text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        spam_score = probs[0][1].item()
        prediction = "Spam" if spam_score > 0.5 else "Not Spam"
    return prediction, spam_score

def main():
    st.set_page_config(page_title="Call Spam Detector", page_icon="📞", layout="centered")
    st.title("📞 Call Spam Detector - BERT")
    st.write("Upload a call recording, and the model will classify whether it's spam or not.")
    
    uploaded_file = st.file_uploader("Upload your call recording (MP3, WAV, OGG, FLAC)", type=["mp3", "wav", "ogg", "flac"])
    
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        
        with st.spinner("Transcribing audio..."):
            text = audio_to_text(uploaded_file)
            
        st.subheader("Transcribed Text:")
        st.write(text)
        
        if text and text not in ["Could not understand the audio.", "Speech-to-text service unavailable."]:
            model, tokenizer, device = load_model()
            with st.spinner("Analyzing the text..."):
                prediction, spam_score = predict_spam(text, model, tokenizer, device)
                
            st.subheader("Prediction:")
            st.write(f"**{prediction}**")
            st.progress(spam_score)
            st.write(f"Spam Confidence Score: {spam_score:.2f}")
        else:
            st.error("Failed to transcribe the audio. Please try another file.")
    
if __name__ == "__main__":
    main()
