"""
Enhanced Indonesian Pronunciation App
with Support for Google Cloud Storage, DataFrame, cahya/wav2vec2-large-xlsr-indonesian, and OpenAI Whisper.
This version reads audio files and sentence data from Google Cloud Storage.
Cleaner UI with changes of scoring calculation.
"""

import sys
import os

# Print the Python path to help with debugging
print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Print current directory
print(f"Current directory: {os.getcwd()}")

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"Added to path: {project_root}")

# Additional debugging
print(f"File exists check - sidebar.py: {os.path.exists(os.path.join(os.path.dirname(__file__), 'interface', 'sidebar.py'))}")

import streamlit as st
import json
from google.oauth2 import service_account
from google.cloud import storage
import warnings
warnings.filterwarnings("ignore")

# Import app modules
from app.interface.sidebar import setup_sidebar
from app.interface.feedback import display_pronunciation_feedback
from app.interface.audio import record_audio
from app.ml_logic.models import load_wav2vec2_model, load_whisper_model, load_epitran
from app.ml_logic.phonemes import ensure_consistent_phoneme_extraction, compare_phonemes, text_to_phonemes
from app.ml_logic.speech import recognize_speech
from app.data.gcs import initialize_gcs_client, load_sentences_dataframe_from_gcs
from app.data.sentences import get_audio_for_sentence, setup_common_voice_data
from app.utils.audio_processing import compare_acoustic_features
from app.utils.text_processing import compare_text_content
from app.utils.translations import translate_text
import app.utils.session_state as session_state

import random

# Set page configuration
st.set_page_config(page_title="Natify: Your Indonesian Pronunciation Coach", page_icon="🇮🇩", layout="wide")

# Create API client for Google Cloud
try:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = storage.Client(credentials=credentials)
    st.success("Connected to Google Cloud Storage successfully")
except Exception as e:
    st.error(f"Failed to connect to Google Cloud: {e}")
    st.warning("Continuing with limited functionality...")

# Fixed GCS configuration - replace with your actual values
GCS_BUCKET_NAME = "natify"  # Replace with your actual bucket name
GCS_TSV_PATH = "final_audio/filtered_results.tsv"  # Replace with your actual TSV file path

# Initialize session states
session_state.initialize_session_state(GCS_BUCKET_NAME, GCS_TSV_PATH)

# Main application
def main():
    # Initialize data
    common_voice_data = setup_common_voice_data()

    # Initialize GCS connection at startup
    if st.session_state.gcs_client is None:
        with st.spinner("Connecting to Google Cloud Storage..."):
            st.session_state.gcs_client = initialize_gcs_client(GCS_BUCKET_NAME)

            if st.session_state.gcs_client:
                # Load the dataframe from GCS
                st.session_state.original_sentences_df = load_sentences_dataframe_from_gcs(
                    GCS_BUCKET_NAME,
                    GCS_TSV_PATH
                )
                # Initialize both DataFrames to the same value initially
                if st.session_state.original_sentences_df is not None:
                    st.session_state.sentences_df = st.session_state.original_sentences_df.copy()
                    st.session_state.filtered_sentences_df = st.session_state.original_sentences_df.copy()
                    st.success(f"Successfully loaded {len(st.session_state.original_sentences_df)} sentences from GCS")

    # UI Components
    st.title("🇮🇩 Natify: Your Indonesian Pronunciation Coach")
    st.markdown("Practice your Indonesian pronunciation with phoneme-level analysis!")

    # Setup sidebar with filters and options
    model_type, stt_model, recording_duration, difficulty = setup_sidebar()

    # Load the selected model
    if model_type == "cahya-indonesian":
        model_name = "cahya/wav2vec2-large-xlsr-indonesian"
    elif model_type == "wav2vec2-lv-60":
        model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    else:
        model_name = "facebook/wav2vec2-large-xlsr-53"

    model, processor, feature_extractor = load_wav2vec2_model(model_name)

    # Load Epitran for text-to-phoneme conversion
    epi = load_epitran()

    # Main content
    col1, col2 = st.columns([3, 2])

    with col1:
        # Generate new sentence button
        if st.button("Get New Sentence"):
            if st.session_state.filtered_sentences_df is not None and not st.session_state.filtered_sentences_df.empty:
                # Select a random sentence from the filtered DataFrame
                random_idx = random.randint(0, len(st.session_state.filtered_sentences_df) - 1)
                sentence_row = st.session_state.filtered_sentences_df.iloc[random_idx]

                st.session_state.current_sentence = sentence_row['sentence']

                # Get translation if available or automatically translate it
                if 'translation' in sentence_row and sentence_row['translation'] != "Translation not available":
                    st.session_state.current_translation = sentence_row['translation']
                else:
                    # Automatically translate without requiring user to press a button
                    with st.spinner("Translating..."):
                        translation = translate_text(st.session_state.current_sentence)
                        st.session_state.current_translation = translation

                # Determine and store current difficulty for the selected sentence
                if 'difficulty' in sentence_row:
                    st.session_state.current_difficulty = sentence_row['difficulty']
                else:
                    # Default to medium if not found
                    st.session_state.current_difficulty = "medium"

                # Update recording duration based on difficulty immediately
                duration_map = {
                    "easy": 2,
                    "medium": 3,
                    "difficult": 4
                }
                st.session_state.recording_duration = duration_map.get(st.session_state.current_difficulty, 3)

                # Generate phonemes for the sentence (used for audio comparison)
                if epi:
                    st.session_state.current_phonemes = text_to_phonemes(st.session_state.current_sentence, epi)
                else:
                    st.session_state.current_phonemes = "Phoneme conversion not available"

                # Generate text-based phonemes for display to user before recording
                if epi and st.session_state.current_sentence:
                    st.session_state.text_phonemes = text_to_phonemes(st.session_state.current_sentence, epi)
                else:
                    st.session_state.text_phonemes = "Phoneme conversion not available"

                # Get audio file path from GCS with explicit bucket name
                audio_path = get_audio_for_sentence(
                    st.session_state.current_sentence,
                    st.session_state.sentences_df,
                    bucket_name=st.session_state.gcs_bucket_name,
                    fallback_to_tts=True
                )

                # Store the path and show audio source
                st.session_state.audio_path = audio_path
            else:
                # Use the default sentences database from sentences.py
                from app.data.sentences import sentences_db

                # Select a random sentence from the chosen difficulty
                sentence_data = random.choice(sentences_db[difficulty])
                st.session_state.current_sentence = sentence_data["sentence"]
                st.session_state.current_translation = sentence_data["translation"]

                # Store the difficulty for recording duration defaults
                st.session_state.current_difficulty = difficulty

                # Update recording duration based on difficulty immediately
                duration_map = {
                    "easy": 2,
                    "medium": 3,
                    "difficult": 4
                }
                st.session_state.recording_duration = duration_map.get(difficulty, 3)

                # Generate phonemes for the sentence (used for audio comparison)
                if epi:
                    st.session_state.current_phonemes = text_to_phonemes(st.session_state.current_sentence, epi)
                else:
                    st.session_state.current_phonemes = "Phoneme conversion not available"

                # Generate text-based phonemes for user display
                if epi and st.session_state.current_sentence:
                    st.session_state.text_phonemes = text_to_phonemes(st.session_state.current_sentence, epi)
                else:
                    st.session_state.text_phonemes = "Phoneme conversion not available"

                # Get audio file path from GCS or fallback to built-in
                audio_path = get_audio_for_sentence(
                    st.session_state.current_sentence,
                    common_voice_data,
                    bucket_name=st.session_state.gcs_bucket_name,
                    fallback_to_tts=True
                )

            # Store the path
            st.session_state.audio_path = audio_path

            # Reset recording and score
            session_state.reset_session_scores()

        # Display current sentence and translation
        if st.session_state.current_sentence:
            st.subheader("Sentence to Pronounce:")
            st.markdown(f"### {st.session_state.current_sentence}")

            # Display translation directly without the translate button
            if st.session_state.current_translation == "Translation not available":
                st.markdown("*Translation: Not available*")
            else:
                st.markdown(f"*Translation: {st.session_state.current_translation}*")

            # Display phonemes
            st.subheader("Phonetic Pronunciation:")
            if 'text_phonemes' in st.session_state and st.session_state.text_phonemes:
                st.markdown("**Expected pronunciation (text-based):**")
                st.markdown(f"`{st.session_state.text_phonemes}`")

            # Only show the audio-derived phonemes if they exist
            if 'current_phonemes' in st.session_state and st.session_state.current_phonemes and st.session_state.user_recording:
                with st.expander("Show audio-derived phoneme fingerprints (technical)"):
                    st.markdown("**Reference audio phonemes:**")
                    st.markdown(f"`{st.session_state.current_phonemes}`")

                    if 'recognized_phonemes' in st.session_state and st.session_state.recognized_phonemes:
                        st.markdown("**Your audio phonemes:**")
                        st.markdown(f"`{st.session_state.recognized_phonemes}`")

                        st.info("These are the actual phoneme fingerprints being compared to calculate your score.")

            # Audio playback
            st.audio(st.session_state.audio_path, format='audio/wav')
            st.markdown("👆 Listen to the pronunciation and try to repeat it")

            # Record user's pronunciation
            if st.button(f"Record Your Pronunciation ({recording_duration} seconds)"):
                user_audio_path = record_audio(duration=recording_duration)
                st.session_state.user_recording = user_audio_path
                st.audio(user_audio_path, format='audio/wav')

                # Process recording if models are loaded
                if st.session_state.audio_path and st.session_state.user_recording:
                    with st.spinner('Analyzing your pronunciation...'):
                        # Extract phonemes using wav2vec2 if models are loaded
                        if model and processor:
                            # Always extract phonemes from audio for both recordings using the same method
                            reference_phonemes, recognized_phonemes = ensure_consistent_phoneme_extraction(
                                st.session_state.audio_path,
                                st.session_state.user_recording,
                                model, processor, feature_extractor
                            )

                            # Store both phoneme sets
                            st.session_state.current_phonemes = reference_phonemes
                            st.session_state.recognized_phonemes = recognized_phonemes

                            # Compare phonemes (now both derived from audio, not text)
                            phoneme_score, comparison = compare_phonemes(
                                reference_phonemes,
                                recognized_phonemes
                            )
                            st.session_state.phoneme_score = phoneme_score
                            st.session_state.phoneme_comparison = comparison

                            # Still keep text-based phonemes for display purposes
                            if epi and st.session_state.current_sentence:
                                st.session_state.text_phonemes = text_to_phonemes(st.session_state.current_sentence, epi)
                            else:
                                st.session_state.text_phonemes = "Text-based phoneme conversion not available"

                        # Get acoustic score
                        acoustic_score = compare_acoustic_features(
                            st.session_state.audio_path,
                            st.session_state.user_recording,
                            st.session_state.recognized_phonemes  # Pass recognized phonemes to the function
                        )
                        st.session_state.acoustic_score = acoustic_score

                        # Perform speech recognition for content comparison using selected model
                        recognized_text = recognize_speech(
                            st.session_state.user_recording,
                            language='id',
                            stt_model=st.session_state.stt_model
                        )
                        st.session_state.recognized_text = recognized_text

                        # Get content score
                        content_score = compare_text_content(
                            st.session_state.current_sentence,
                            recognized_text
                        )
                        st.session_state.content_score = content_score

                        # Calculate final score - weighted combination
                        if st.session_state.phoneme_score is not None:
                            final_score = (
                                (acoustic_score * 0.3) +
                                (content_score * 0.3) +
                                (st.session_state.phoneme_score * 0.4)
                            )
                        else:
                            final_score = (acoustic_score * 0.4) + (content_score * 0.6)

                        st.session_state.score = final_score

                        # Update progress tracking - only count each new recording once
                        if st.session_state.score != st.session_state.last_recorded_score:
                            st.session_state.total_attempts += 1

                            # Determine difficulty
                            current_difficulty = "easy"
                            if st.session_state.sentences_df is not None:
                                # Try to find the sentence in the DataFrame to get its difficulty
                                sentence_match = st.session_state.sentences_df[
                                    st.session_state.sentences_df['sentence'] == st.session_state.current_sentence
                                ]
                                if not sentence_match.empty and 'difficulty' in sentence_match.columns:
                                    current_difficulty = sentence_match.iloc[0]['difficulty']

                                    # Update recording duration based on sentence difficulty
                                    duration_map = {
                                        "easy": 2,
                                        "medium": 3,
                                        "difficult": 4
                                    }
                                    st.session_state.recording_duration = duration_map.get(current_difficulty, 3)
                            else:
                                # Use the selected difficulty from the sidebar
                                current_difficulty = difficulty

                            st.session_state.difficulty_attempts[current_difficulty] += 1
                            if st.session_state.score >= 70:
                                st.session_state.successful_attempts += 1
                                st.session_state.difficulty_success[current_difficulty] += 1
                            st.session_state.last_recorded_score = st.session_state.score
        else:
            st.info("Click 'Get New Sentence' to start practicing!")

    with col2:
        # Show results and feedback
        if st.session_state.score is not None:
            st.subheader("Pronunciation Score")

            # Display score with color
            score = st.session_state.score
            color = "green" if score >= 80 else "orange" if score >= 60 else "red"
            st.markdown(f"<h1 style='color:{color}'>{score:.1f}%</h1>", unsafe_allow_html=True)

            # Display component scores
            st.write(f"Acoustic Score: {st.session_state.acoustic_score:.1f}%")
            st.write(f"Content Score: {st.session_state.content_score:.1f}%")
            if st.session_state.phoneme_score is not None:
                st.write(f"Phoneme Score: {st.session_state.phoneme_score:.1f}%")

            # Display which models were used
            st.write(f"STT Model: {st.session_state.stt_model.capitalize()}")
            phoneme_model = "Cahya Indonesian" if st.session_state.model_type == "cahya-indonesian" else st.session_state.model_type
            st.write(f"Phoneme Model: {phoneme_model}")

            # Provide feedback based on score
            if score >= 80:
                st.success("Excellent pronunciation! Keep it up! 👏")
            elif score >= 60:
                st.warning("Good attempt! Try to match the phonemes and rhythm more closely.")
            else:
                st.error("Keep practicing! Listen carefully to the original audio and try again.")

            # Show what was recognized
            st.subheader("What We Heard:")
            if st.session_state.recognized_text:
                st.write(f'"{st.session_state.recognized_text}"')
            else:
                st.write("We couldn't detect any speech. Please try again and speak clearly.")

            # Use our simplified audio analysis function
            if st.session_state.user_recording:
                display_pronunciation_feedback(
                    st.session_state.user_recording,
                    st.session_state.audio_path,
                    st.session_state.phoneme_comparison,
                    st.session_state.recognized_text,
                    st.session_state.current_sentence
                )

if __name__ == "__main__":
    main()
