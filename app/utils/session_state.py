"""
Session state management for the Indonesian Pronunciation App.
"""

import streamlit as st

def initialize_session_state(gcs_bucket_name, gcs_tsv_path):
    """
    Initialize all session state variables.

    Args:
        gcs_bucket_name: Name of the GCS bucket
        gcs_tsv_path: Path to the TSV file in the bucket
    """
    # Initialize session states
    if 'current_sentence' not in st.session_state:
        st.session_state.current_sentence = ""
    if 'current_translation' not in st.session_state:
        st.session_state.current_translation = ""
    if 'current_phonemes' not in st.session_state:
        st.session_state.current_phonemes = ""
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None
    if 'user_recording' not in st.session_state:
        st.session_state.user_recording = None
    if 'score' not in st.session_state:
        st.session_state.score = None
    if 'phoneme_score' not in st.session_state:
        st.session_state.phoneme_score = None
    if 'acoustic_score' not in st.session_state:
        st.session_state.acoustic_score = None
    if 'content_score' not in st.session_state:
        st.session_state.content_score = None
    if 'recognized_text' not in st.session_state:
        st.session_state.recognized_text = None
    if 'recognized_phonemes' not in st.session_state:
        st.session_state.recognized_phonemes = None
    if 'common_voice_data' not in st.session_state:
        st.session_state.common_voice_data = None
    if 'total_attempts' not in st.session_state:
        st.session_state.total_attempts = 0
    if 'successful_attempts' not in st.session_state:
        st.session_state.successful_attempts = 0
    if 'difficulty_attempts' not in st.session_state:
        st.session_state.difficulty_attempts = {"easy": 0, "medium": 0, "difficult": 0}
    if 'difficulty_success' not in st.session_state:
        st.session_state.difficulty_success = {"easy": 0, "medium": 0, "difficult": 0}
    if 'last_recorded_score' not in st.session_state:
        st.session_state.last_recorded_score = None
    if 'phoneme_comparison' not in st.session_state:
        st.session_state.phoneme_comparison = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = "cahya-indonesian"
    if 'stt_model' not in st.session_state:
        st.session_state.stt_model = "whisper"
    if 'is_using_tts' not in st.session_state:
        st.session_state.is_using_tts = False
    if 'sentences_df' not in st.session_state:
        st.session_state.sentences_df = None
    if 'original_sentences_df' not in st.session_state:
        st.session_state.original_sentences_df = None
    if 'filtered_sentences_df' not in st.session_state:
        st.session_state.filtered_sentences_df = None
    if 'original_filename' not in st.session_state:
        st.session_state.original_filename = None
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = None
    if 'gcs_client' not in st.session_state:
        st.session_state.gcs_client = None
    if 'gcs_bucket_name' not in st.session_state:
        st.session_state.gcs_bucket_name = gcs_bucket_name
    if 'gcs_tsv_path' not in st.session_state:
        st.session_state.gcs_tsv_path = gcs_tsv_path
    if 'text_phonemes' not in st.session_state:
        st.session_state.text_phonemes = None
    if 'current_difficulty' not in st.session_state:
        st.session_state.current_difficulty = "medium"
    if 'recording_duration' not in st.session_state:
        st.session_state.recording_duration = 3
    if 'input_device_id' not in st.session_state:
        st.session_state.input_device_id = None

def reset_session_scores():
    """
    Reset only the scoring-related session state variables.
    """
    st.session_state.user_recording = None
    st.session_state.score = None
    st.session_state.acoustic_score = None
    st.session_state.content_score = None
    st.session_state.phoneme_score = None
    st.session_state.recognized_text = None
    st.session_state.recognized_phonemes = None
    st.session_state.phoneme_comparison = None
    st.session_state.last_recorded_score = None  # Reset this to ensure new attempts are counted
