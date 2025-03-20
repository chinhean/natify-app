"""
Audio recording and playback components for the Indonesian Pronunciation App.
"""

import streamlit as st
import sounddevice as sd
import soundfile as sf
import time
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa

def list_audio_devices():
    """
    List all available audio input devices.

    Returns:
        List of audio input devices
    """
    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            # Format device info for display
            input_devices.append({
                'id': i,
                'name': device['name'],
                'channels': device['max_input_channels']
            })

    return input_devices

def record_audio(duration=3, sample_rate=16000):
    """
    Record audio from the user's microphone with selected device

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        String: Path to the recorded audio file
    """
    # Get the selected input device ID from session state
    device_id = st.session_state.get('input_device_id', None)

    st.write("🎙️ Recording... Speak now!")

    # Start recording with the selected device
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        device=device_id  # Use the selected device
    )

    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(duration/100)
        progress_bar.progress(i + 1)
    sd.wait()
    progress_bar.empty()

    # Save the recording to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, recording, sample_rate)
    return temp_file.name

def plot_waveform(audio_path):
    """
    Plot the waveform of an audio file

    Args:
        audio_path: Path to the audio file

    Returns:
        Figure: Matplotlib figure with the waveform
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)  # Always use 16000 Hz
        plt.figure(figsize=(10, 2))
        plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
        plt.title('Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        fig = plt.gcf()
        plt.close()
        return fig
    except Exception as e:
        st.error(f"Error plotting waveform: {e}")
        # Return a simple figure with error message
        fig = plt.figure(figsize=(10, 2))
        plt.text(0.5, 0.5, f"Error generating waveform: {str(e)}",
                 horizontalalignment='center', verticalalignment='center')
        plt.close()
        return fig

def display_audio_info(audio_path, target_sr=16000):
    """
    Display audio information with consistent sample rate.

    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate for analysis (default: 16000)

    Returns:
        Dictionary with audio information
    """
    try:
        # Always resample to the consistent sample rate for comparison
        y, _ = librosa.load(audio_path, sr=target_sr)
        duration = librosa.get_duration(y=y, sr=target_sr)
        return {
            "duration": f"{duration:.2f} seconds",
            "sample_rate": f"{target_sr} Hz",
            "num_samples": len(y),
            "max_amplitude": f"{np.max(np.abs(y)):.4f}",
            "mean_amplitude": f"{np.mean(np.abs(y)):.4f}"
        }
    except Exception as e:
        return {"error": str(e)}

def plot_phoneme_comparison(expected_phonemes, recognized_phonemes, comparison):
    """
    Create a visualization of phoneme comparison

    Args:
        expected_phonemes: Expected phonemes
        recognized_phonemes: Recognized phonemes
        comparison: Comparison data

    Returns:
        Figure: Matplotlib figure with phoneme comparison
    """
    try:
        # Create a figure
        fig = plt.figure(figsize=(10, 3))
        plt.axis('off')  # Hide axes
        plt.title('Phoneme Comparison: Expected vs. Actual', fontsize=14)

        # Define better colors for each match type
        colors = {
            "match": "#c6efce",    # Light green
            "replace": "#ffeb9c",  # Light yellow
            "delete": "#ffc7ce",   # Light red
            "insert": "#b7dee8",   # Light blue
            "perfect": "#c6efce"   # Same as match (light green)
        }

        # Define labels for the legend
        labels = {
            "match": "Match",
            "replace": "Different",
            "delete": "Missing",
            "insert": "Extra",
            "perfect": "Perfect Match"
        }

        # Create visual elements for the legend (excluding "perfect" to avoid duplicate)
        legend_types = ["match", "replace", "delete", "insert"]
        for match_type in legend_types:
            plt.bar(0, 0, color=colors[match_type], label=labels[match_type], alpha=0.7)

        # Add legend above the comparison table
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=4, frameon=False)

        # Prepare data for table
        expected_row = []
        recognized_row = []
        cell_colors = []

        # Process comparison data
        for match_type, exp, rec in comparison:
            # If it's a perfect match, treat it as a regular match for display
            if match_type == "perfect":
                match_type = "match"

            expected_row.append(exp if exp else "-")
            recognized_row.append(rec if rec else "-")
            cell_colors.append(colors[match_type])

        # Create the table with headers
        cell_text = [expected_row, recognized_row]

        # Generate table with colored cells
        table = plt.table(
            cellText=cell_text,
            rowLabels=['Expected', 'Actual'],
            loc='center',
            cellLoc='center',
            cellColours=[['white'] * len(cell_colors), cell_colors]
        )

        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating phoneme visualization: {e}")
        # Return a simple figure with error message
        fig = plt.figure(figsize=(10, 3))
        plt.text(0.5, 0.5, f"Error generating phoneme comparison: {str(e)}",
                 horizontalalignment='center', verticalalignment='center')
        plt.close()
        return fig

def display_audio_analysis(user_recording, reference_recording):
    """
    Display audio analysis information

    Args:
        user_recording: Path to user's recording
        reference_recording: Path to reference recording
    """
    st.subheader("Audio Analysis")

    # Show simplified audio information
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("Your Recording:")
        user_audio_info = display_audio_info(user_recording)
        for key, value in user_audio_info.items():
            st.write(f"- {key}: {value}")

    with col_b:
        st.write("Reference Audio:")
        ref_audio_info = display_audio_info(reference_recording)
        for key, value in ref_audio_info.items():
            st.write(f"- {key}: {value}")
