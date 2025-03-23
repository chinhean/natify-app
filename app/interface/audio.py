"""
Audio recording and playback components for the Indonesian Pronunciation App.
Using Streamlit's built-in audio recorder with fallback to sounddevice.
"""

import streamlit as st
import sounddevice as sd
import soundfile as sf
import time
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import traceback
import importlib
import sys

def list_audio_devices():
    """
    List all available audio input devices.
    In cloud deployment, this will return a mock device list.

    Returns:
        List of audio input devices
    """
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
    except Exception as e:
        # If sounddevice fails, return a mock device
        input_devices = [{
            'id': 0,
            'name': 'Default Microphone',
            'channels': 1
        }]

    # If no devices found, add a default option
    if not input_devices:
        input_devices = [{
            'id': 0,
            'name': 'Default Microphone',
            'channels': 1
        }]

    return input_devices

def record_audio(duration=3, sample_rate=16000):
    """
    Record audio using Streamlit's built-in audio_recorder when available,
    with fallback to sounddevice.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        String: Path to the recorded audio file
    """
    # Try to use Streamlit's built-in audio recorder first
    has_audio_recorder = hasattr(st, 'audio_recorder')

    if has_audio_recorder:
        try:
            return _record_audio_streamlit(duration, sample_rate)
        except Exception as e:
            st.warning(f"Streamlit audio recorder failed: {str(e)}. Falling back to sounddevice.")
            return _record_audio_sounddevice(duration, sample_rate)
    else:
        st.info("Using native recording function. For cloud deployment, consider upgrading Streamlit.")
        return _record_audio_sounddevice(duration, sample_rate)

def _record_audio_streamlit(duration=3, sample_rate=16000):
    """
    Record audio using Streamlit's built-in audio_recorder.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        String: Path to the recorded audio file
    """
    # Create UI elements
    status_container = st.empty()
    status_container.info(f"Please record for approximately {duration} seconds.")

    # Use Streamlit's audio recorder
    audio_bytes = st.audio_recorder(pause_threshold=duration+1.0)

    if audio_bytes is not None:
        status_container.success("Recording received. Processing audio...")

        # Save audio bytes to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(audio_bytes)
        temp_file.close()

        # Process the audio to ensure correct format and duration
        processed_file = _process_audio_file(temp_file.name, duration, sample_rate)

        # Clean up original temp file if needed
        if processed_file != temp_file.name and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass

        status_container.empty()
        return processed_file
    else:
        status_container.warning("No audio recorded. Please try again or use alternative method.")
        status_container.empty()
        # Fall back to sounddevice if no audio was captured
        return _record_audio_sounddevice(duration, sample_rate)

def _record_audio_sounddevice(duration=3, sample_rate=16000):
    """
    Record audio using sounddevice with improved error handling.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        String: Path to the recorded audio file
    """
    # Ensure duration is a reasonable value
    duration = max(1, min(10, duration))  # Between 1 and 10 seconds

    # Create UI elements
    record_container = st.empty()
    progress_container = st.empty()
    status_container = st.empty()

    try:
        # Get the selected input device ID from session state
        device_id = st.session_state.get('input_device_id', None)

        # Get device list and try to validate the selected device
        devices = list_audio_devices()
        valid_device = False
        device_name = "Default"

        for device in devices:
            if device['id'] == device_id:
                valid_device = True
                device_name = device['name']
                break

        if not valid_device:
            status_container.warning("Selected audio device not found, using default.")
            device_id = None

        # Display recording status
        record_container.markdown(f"🎙️ Recording from **{device_name}**... Please speak clearly!")

        # Initialize progress bar
        progress_bar = progress_container.progress(0)

        # Start recording with exception handling
        try:
            # First check if sounddevice can initialize
            sd.check_input_settings(
                device=device_id,
                channels=1,
                samplerate=sample_rate,
                dtype='float32'
            )

            # Start the recording
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                device=device_id  # Use the selected device or default
            )

            # Update progress bar
            for i in range(100):
                time.sleep(duration/100)
                progress_bar.progress(i + 1)
                remaining = duration - (i+1) * duration/100
                status_container.text(f"Recording... {remaining:.1f} seconds remaining.")

            # Ensure recording is complete
            sd.wait()
            status_container.success("Recording completed successfully!")

            # Check if recording contains actual sound data
            if np.max(np.abs(recording)) < 0.01:
                status_container.warning("Recording seems too quiet. Please speak louder next time.")

            # Save the recording to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, recording, sample_rate)
            return temp_file.name

        except Exception as e:
            # Handle recording errors
            error_msg = f"Error during recording: {str(e)}"
            status_container.error(error_msg)

            # Create a silent audio file as fallback
            status_container.warning("Creating silent audio as fallback...")
            return _create_silent_audio(duration, sample_rate)

    except Exception as e:
        # Handle any other errors
        status_container.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return _create_silent_audio(duration, sample_rate)
    finally:
        # Clean up UI elements
        time.sleep(1)
        record_container.empty()
        progress_container.empty()
        status_container.empty()

def _process_audio_file(audio_path, target_duration, target_sr=16000):
    """
    Process an audio file to ensure it meets the required specifications:
    - Has the correct sample rate
    - Is approximately the target duration (trimming or padding as needed)
    - Is in mono format

    Args:
        audio_path: Path to the audio file to process
        target_duration: Target duration in seconds
        target_sr: Target sample rate

    Returns:
        Path to the processed audio file
    """
    try:
        # Load the audio
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Calculate current duration
        current_duration = len(y) / target_sr

        # Adjust duration
        if current_duration > target_duration * 1.5:  # If much longer, trim to exact duration
            # Trim to exact target duration
            y = y[:int(target_duration * target_sr)]
        elif current_duration < target_duration * 0.5:  # If much shorter, pad with silence
            # Pad with silence to reach target duration
            padding = np.zeros(int(target_sr * target_duration) - len(y))
            y = np.concatenate((y, padding))

        # Save to a new temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()
        sf.write(temp_file.name, y, target_sr)

        return temp_file.name
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        # Return the original file if processing fails
        return audio_path

def _create_silent_audio(duration, sample_rate):
    """Create a silent audio file as a fallback"""
    silent_audio = np.zeros(int(sample_rate * duration))
    silent_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(silent_file.name, silent_audio, sample_rate)
    return silent_file.name

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
