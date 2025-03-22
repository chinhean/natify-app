"""
Audio recording and playback components for the Indonesian Pronunciation App.
With automatic detection for cloud vs local environment.
"""

import streamlit as st
import soundfile as sf
import time
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

# Detect if running on Streamlit Cloud
# When on Cloud, servers usually listen on 0.0.0.0 address
is_cloud = os.environ.get('STREAMLIT_SERVER_ADDRESS', '').startswith('0.0.0.0')
# Additional cloud indicators
is_cloud = is_cloud or 'STREAMLIT_CLOUD' in os.environ or 'STREAMLIT_SHARING' in os.environ

# Import streamlit-audiorec if available
try:
    from streamlit_audiorec import st_audiorec
    audiorec_available = True
except ImportError:
    audiorec_available = False

# Import sounddevice only for local environments
if not is_cloud:
    try:
        import sounddevice as sd
    except ImportError:
        sd = None
else:
    sd = None

def list_audio_devices():
    """
    List all available audio input devices.
    Will only work in local environments.

    Returns:
        List of audio input devices or empty list if on cloud
    """
    if is_cloud or sd is None:
        return []

    try:
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
    except Exception as e:
        st.error(f"Error listing audio devices: {e}")
        return []

def record_audio(duration=3, sample_rate=16000):
    """
    Smart record audio function that automatically chooses the appropriate method.
    In local environments, uses sounddevice to access the microphone directly.
    In cloud environments, uses streamlit-audiorec for browser-based recording.

    Args:
        duration: Recording duration in seconds (used only for device recording)
        sample_rate: Sample rate in Hz

    Returns:
        String: Path to the recorded audio file or None if error
    """
    # For cloud environments, use browser-based recording
    if is_cloud:
        if not audiorec_available:
            st.error("Cloud environment detected but streamlit-audiorec is not installed.")
            st.info("Please add 'streamlit-audiorec==0.1.3' to your requirements.txt file.")
            return None

        return record_audio_browser_internal()

    # For local environments, use device-based recording
    else:
        if sd is None:
            st.error("sounddevice module is not available for local recording.")
            return None

        return record_audio_device_internal(duration, sample_rate)

def record_audio_device_internal(duration=3, sample_rate=16000):
    """
    Internal function to record audio from the user's microphone with selected device.
    Works only in local environments.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        String: Path to the recorded audio file or None if error
    """
    try:
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
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

def record_audio_browser_internal():
    """
    Internal function to record audio using the browser's microphone via streamlit-audiorec.
    Works in all environments, including cloud.

    Returns:
        String: Path to the recorded audio file or None if no recording
    """
    st.write("🎙️ Click below to record your voice")

    # Use the streamlit-audiorec component to record audio
    audio_bytes = st_audiorec()

    # If audio was recorded, save it to a file
    if audio_bytes is not None:
        # Save the recording to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(audio_bytes)
        temp_file.close()

        # Normalize to expected format (16kHz)
        try:
            y, sr = librosa.load(temp_file.name)
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sf.write(temp_file.name, y, 16000)
        except Exception as e:
            st.error(f"Error processing audio: {e}")

        return temp_file.name

    return None

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
