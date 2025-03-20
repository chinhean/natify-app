"""
Speech recognition functionality for the Indonesian Pronunciation App.
"""

import streamlit as st
import speech_recognition as sr

def recognize_speech(audio_path, language='id', stt_model="whisper"):
    """
    Perform speech recognition on audio using selected model.

    Args:
        audio_path: Path to the audio file
        language: Language code (default: 'id' for Indonesian)
        stt_model: Speech recognition model to use ('whisper' or 'google')

    Returns:
        String: Recognized text
    """
    try:
        if stt_model == "whisper":
            # Use OpenAI's Whisper model
            if st.session_state.whisper_model is None:
                from app.ml_logic.models import load_whisper_model
                whisper_size = st.session_state.get('whisper_size', 'base')
                st.session_state.whisper_model = load_whisper_model(whisper_size)

            if st.session_state.whisper_model:
                result = st.session_state.whisper_model.transcribe(
                    audio_path,
                    language=language,
                    fp16=False  # Disable fp16 for better compatibility
                )
                return result["text"].lower()
            else:
                st.error("Whisper model is not loaded. Falling back to Google STT.")
                # Fall back to Google's speech recognition service
                return recognize_speech(audio_path, f"{language}-{language.upper()}", "google")

        elif stt_model == "google":
            # Use Google's speech recognition service
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language=f"{language}-{language.upper()}")
                return text.lower()
        else:
            st.error(f"Unknown STT model: {stt_model}")
            return ""

    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from speech recognition service: {e}")
        return ""
    except Exception as e:
        st.error(f"Error in speech recognition: {e}")
        return ""
