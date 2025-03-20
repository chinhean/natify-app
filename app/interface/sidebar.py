"""
Sidebar components for the Indonesian Pronunciation App.
"""

import streamlit as st

def setup_sidebar():
    """
    Set up the sidebar with filter options and model selections.

    Returns:
        Tuple: (model_type, stt_model, recording_duration, difficulty)
    """
    st.sidebar.header("Settings")

    # Only show difficulty selection if DataFrame is loaded from GCS
    if st.session_state.original_sentences_df is not None:
        # Filter by difficulty only - single selection
        st.sidebar.subheader("Filter Difficulty")
        selected_difficulty = st.sidebar.radio(
            "Select difficulty level:",
            ["easy", "medium", "difficult"],
            index=0  # Default to "easy"
        )

        # Filter DataFrame by selected difficulty
        # But keep the original_sentences_df intact
        filtered_df = st.session_state.original_sentences_df.copy()
        filtered_df = filtered_df[filtered_df['difficulty'] == selected_difficulty]

        # Map difficulty to duration
        duration_map = {
            "easy": 2,
            "medium": 3,
            "difficult": 4
        }

        # Update session state values
        st.session_state.current_difficulty = selected_difficulty
        st.session_state.recording_duration = duration_map.get(selected_difficulty, 3)

        # Only update the filtered_sentences_df, not the original
        st.session_state.filtered_sentences_df = filtered_df
        st.sidebar.info(f"Showing {len(filtered_df)} sentences with {selected_difficulty} difficulty")

    # If no DataFrame is loaded, show built-in difficulty selection
    else:
        # Show the difficulty selection only when using built-in sentences
        difficulty = st.sidebar.radio("Choose Difficulty Level (for built-in sentences):",
                                     ["easy", "medium", "difficult"])

        # Update current_difficulty immediately when the user changes it in the sidebar
        if 'current_difficulty' not in st.session_state or st.session_state.current_difficulty != difficulty:
            st.session_state.current_difficulty = difficulty

            # Update recording duration based on new difficulty
            duration_map = {
                "easy": 2,
                "medium": 3,
                "difficult": 4
            }
            st.session_state.recording_duration = duration_map.get(difficulty, 3)

        st.sidebar.markdown("---")  # Add separator for visual clarity

    # Set a default difficulty level that will be used only as fallback
    difficulty = "medium"

    # Initialize current difficulty if not already set
    if 'current_difficulty' not in st.session_state:
        st.session_state.current_difficulty = "medium"  # Default to medium if not set

    # Map difficulty to duration
    duration_map = {
        "easy": 2,
        "medium": 3,
        "difficult": 4
    }

    # Use session state for recording duration, with difficulty-based default if not set
    if 'recording_duration' not in st.session_state:
        st.session_state.recording_duration = duration_map.get(st.session_state.current_difficulty, 3)

    # Display the recording duration slider with current session state value
    recording_duration = st.sidebar.slider(
        "Recording Duration (seconds)",
        min_value=2,
        max_value=6,
        value=st.session_state.recording_duration,
        help=f"Default duration for {st.session_state.current_difficulty} difficulty is {duration_map.get(st.session_state.current_difficulty, 3)} seconds"
    )

    # Update session state if slider value changes
    if recording_duration != st.session_state.recording_duration:
        st.session_state.recording_duration = recording_duration

    # Add an explanatory note about duration defaults
    st.sidebar.caption(f"""
    Duration settings by difficulty:
    • Easy: 2 seconds
    • Medium: 3 seconds
    • Difficult: 4 seconds
    """)

    # Model selection for phoneme recognition
    st.sidebar.subheader("Phoneme Recognition Model")
    model_type = st.sidebar.radio(
        "Select Phoneme Recognition Model:",
        ["cahya-indonesian", "wav2vec2-lv-60", "wav2vec2-xlsr-53"]
    )

    if model_type != st.session_state.model_type:
        st.session_state.model_type = model_type

    # Speech recognition model selection
    st.sidebar.subheader("Speech Recognition Model")
    stt_model = st.sidebar.radio(
        "Select Speech Recognition Model:",
        ["whisper", "google"]
    )

    if stt_model != st.session_state.stt_model:
        st.session_state.stt_model = stt_model

    # Whisper model size selection (if Whisper is selected)
    if stt_model == "whisper":
        show_whisper_options = st.sidebar.checkbox("Show Whisper Model Options", value=False)

        if show_whisper_options:
            whisper_size = st.sidebar.select_slider(
                "Whisper Model Size:",
                options=["tiny", "base", "small", "medium", "large"],
                value=st.session_state.get('whisper_size', 'base')
            )

            if 'whisper_size' not in st.session_state or whisper_size != st.session_state.whisper_size:
                st.session_state.whisper_size = whisper_size
                # Clear the model to force reload with new size
                st.session_state.whisper_model = None

    # Add microphone selection section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Microphone Selection")

    from app.interface.audio import list_audio_devices
    input_devices = list_audio_devices()

    if not input_devices:
        st.sidebar.warning("No audio input devices detected")
    else:
        # Create a list of device names for the dropdown
        device_names = [f"{d['name']} ({d['channels']} ch)" for d in input_devices]

        # Add a device selection dropdown
        selected_device_idx = st.sidebar.selectbox(
            "Select input microphone:",
            range(len(device_names)),
            format_func=lambda i: device_names[i]
        )

        # Store the selected device ID in session state
        st.session_state.input_device_id = input_devices[selected_device_idx]['id']

        st.sidebar.info(f"Using microphone: {input_devices[selected_device_idx]['name']}")

    # Display progress tracking stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("Progress Tracking")

    # Display stats
    progress_percentage = 0
    if st.session_state.total_attempts > 0:
        progress_percentage = (st.session_state.successful_attempts / st.session_state.total_attempts) * 100

    st.sidebar.write(f"Total Attempts: {st.session_state.total_attempts}")
    st.sidebar.write(f"Successful Attempts: {st.session_state.successful_attempts}")
    st.sidebar.progress(int(progress_percentage))

    # Display difficulty-specific stats
    st.sidebar.subheader("Performance by Difficulty")
    for diff in ["easy", "medium", "difficult"]:
        attempts = st.session_state.difficulty_attempts[diff]
        if attempts > 0:
            success_rate = (st.session_state.difficulty_success[diff] / attempts) * 100
            st.sidebar.write(f"{diff.capitalize()}: {success_rate:.1f}% success ({st.session_state.difficulty_success[diff]}/{attempts})")
        else:
            st.sidebar.write(f"{diff.capitalize()}: No attempts yet")

    # Model Information section with dynamic content
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")

    # Create dynamic descriptions for phoneme recognition models
    phoneme_descriptions = {
        "cahya-indonesian": """
        **Phoneme Recognition: Cahya Indonesian**

        This model specializes in Indonesian language sounds. It analyzes your speech
        to identify how closely your pronunciation matches native Indonesian sounds,
        with particular attention to unique Indonesian phonemes.
        """,

        "wav2vec2-lv-60": """
        **Phoneme Recognition: Wav2vec2 LV-60**

        This versatile model recognizes speech sounds across many languages. It compares
        your pronunciation to standard Indonesian sound patterns and helps identify
        where your pronunciation might differ from native speakers.
        """,

        "wav2vec2-xlsr-53": """
        **Phoneme Recognition: Wav2vec2 XLSR-53**

        This cross-language model is designed to understand speech across multiple
        languages. It analyzes the phonetic components of your speech to identify how
        closely your pronunciation matches the expected Indonesian sounds.
        """
    }

    # Create dynamic descriptions for speech recognition models
    whisper_descriptions = {
        "tiny": """
        **Speech Recognition: Whisper (Tiny)**

        This lightweight model converts your speech to text quickly. It works best
        with clear speech in quiet environments and helps check if the words you're
        saying match the Indonesian text.
        """,

        "base": """
        **Speech Recognition: Whisper (Base)**

        This balanced model provides good speech-to-text conversion for most practice
        situations. It helps identify whether you're saying the correct Indonesian
        words with reasonable accuracy.
        """,

        "small": """
        **Speech Recognition: Whisper (Small)**

        This enhanced model offers improved accuracy in converting your speech to text.
        It's better at understanding varied pronunciations and can handle some
        background noise.
        """,

        "medium": """
        **Speech Recognition: Whisper (Medium)**

        This advanced model provides high accuracy in speech recognition. It's good at
        understanding different accents and speaking styles, helping you get more
        precise feedback on your Indonesian pronunciation.
        """,

        "large": """
        **Speech Recognition: Whisper (Large)**

        This comprehensive model offers maximum accuracy in speech recognition. It
        excels at understanding a wide range of pronunciations and speaking styles,
        providing highly accurate text conversion of your Indonesian practice.
        """
    }

    google_description = """
    **Speech Recognition: Google**

    This cloud-based service converts your speech to text using Google's technology.
    It's particularly good at recognizing clear speech and helps check if the
    Indonesian words you're saying match the expected text.
    """

    # Display the appropriate phoneme model description
    st.sidebar.markdown(phoneme_descriptions[st.session_state.model_type])

    # Display the appropriate speech recognition model description
    if st.session_state.stt_model == "whisper":
        whisper_size = st.session_state.get('whisper_size', 'base')
        st.sidebar.markdown(whisper_descriptions[whisper_size])
    else:
        st.sidebar.markdown(google_description)

    # Simplified About This App section (static content)
    st.sidebar.markdown("---")
    st.sidebar.subheader("About Natify")

    about_text = """
    Natify helps you learn Indonesian pronunciation through practice and personalized feedback.

    **Key Features:**\n
    • Listen to native Indonesian speakers\n
    • Record your own pronunciation attempts\n
    • Get immediate feedback and scoring\n
    • Track your progress over time\n
    • Practice at different difficulty levels\n
    • Use your own custom sentence collections from Google Cloud Storage

    Adjust the settings above to customize your learning experience.
    """

    st.sidebar.info(about_text)

    # Add a small attribution/version
    st.sidebar.caption("Natify - Your Indonesian Pronunciation Coach")

    return model_type, stt_model, recording_duration, difficulty
