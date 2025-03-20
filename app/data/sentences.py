"""
Sentence data and audio management for the Indonesian Pronunciation App.
"""

import streamlit as st
import os
import tempfile
import librosa
import soundfile as sf
from gtts import gTTS
from app.data.gcs import download_blob_to_temp

# Create a database of Indonesian sentences with different difficulty levels
sentences_db = {
    "easy": [
        {"sentence": "Selamat pagi", "translation": "Good morning"},
        {"sentence": "Aku sakit", "translation": "I am sick"},
        {"sentence": "Apa kabar?", "translation": "How are you?"},
        {"sentence": "Terima kasih", "translation": "Thank you"},
        {"sentence": "Nama saya John", "translation": "My name is John"},
        {"sentence": "Saya suka Indonesia", "translation": "I like Indonesia"},
        {"sentence": "Sampai jumpa besok", "translation": "See you tomorrow"},
        {"sentence": "Berapa harganya?", "translation": "How much is it?"},
        {"sentence": "Di mana toilet?", "translation": "Where is the toilet?"},
        {"sentence": "Saya lapar", "translation": "I am hungry"},
        {"sentence": "Selamat malam", "translation": "Good evening"},
        {"sentence": "Ini enak", "translation": "This is delicious"},
        {"sentence": "Saya tidak mengerti", "translation": "I don't understand"},
    ],
    "medium": [
        {"sentence": "Berapa harga makanan ini?", "translation": "How much is this food?"},
        {"sentence": "Dia tidak sepenuhnya mempercayaiku", "translation": "He do not fully trust me"},
        {"sentence": "Di mana stasiun kereta api?", "translation": "Where is the train station?"},
        {"sentence": "Saya belajar bahasa Indonesia", "translation": "I am learning Indonesian"},
        {"sentence": "Hari ini cuaca bagus", "translation": "Today the weather is good"},
        {"sentence": "Boleh saya pesan kopi?", "translation": "May I order coffee?"},
        {"sentence": "Saya tinggal di Jakarta", "translation": "I live in Jakarta"},
        {"sentence": "Jam berapa sekarang?", "translation": "What time is it now?"},
        {"sentence": "Kapan kita akan bertemu?", "translation": "When will we meet?"},
        {"sentence": "Saya suka makanan pedas", "translation": "I like spicy food"},
        {"sentence": "Apakah Anda bisa berbahasa Inggris?", "translation": "Can you speak English?"},
        {"sentence": "Berapa lama Anda tinggal di sini?", "translation": "How long have you been living here?"},
        {"sentence": "Saya perlu membeli tiket", "translation": "I need to buy a ticket"},
    ],
    "difficult": [
        {"sentence": "Keanekaragaman budaya Indonesia sangat menarik",
         "translation": "Indonesia's cultural diversity is very interesting"},
        {"sentence": "Bahasa Indonesia adalah bahasa pemersatu",
         "translation": "Indonesian is a unifying language"},
        {"sentence": "Saya ingin mengunjungi Pulau Komodo tahun depan",
         "translation": "I want to visit Komodo Island next year"},
        {"sentence": "Pembelajaran jarak jauh sangat menantang",
         "translation": "Distance learning is very challenging"},
        {"sentence": "Makanan Indonesia terkenal dengan rempah-rempahnya",
         "translation": "Indonesian food is famous for its spices"},
        {"sentence": "Kebijakan pemerintah tentang pariwisata sedang diperbarui",
         "translation": "Government policies on tourism are being updated"},
        {"sentence": "Perkembangan teknologi digital mengubah cara hidup kita",
         "translation": "Digital technology development is changing our way of life"},
        {"sentence": "Melestarikan budaya lokal sangat penting di era globalisasi",
         "translation": "Preserving local culture is very important in the globalization era"},
        {"sentence": "Kami sedang menghadapi tantangan ekonomi yang signifikan",
         "translation": "We are facing significant economic challenges"},
        {"sentence": "Perubahan iklim mempengaruhi pola tanam petani",
         "translation": "Climate change affects farmers' planting patterns"},
        {"sentence": "Pendidikan berkualitas adalah hak setiap warga negara",
         "translation": "Quality education is the right of every citizen"},
        {"sentence": "Industri kreatif di Indonesia berkembang pesat akhir-akhir ini",
         "translation": "The creative industry in Indonesia has been growing rapidly lately"},
    ]
}

def setup_common_voice_data():
    """
    Simplified function to provide basic structure for Common Voice data.
    In a real implementation, this would connect to the Mozilla Common Voice dataset.
    """
    return {
        "metadata": {
            "language": "id",
            "version": "simplified"
        },
        "clips": []  # We don't need to populate this as we're using TTS
    }

def convert_mp3_to_wav(mp3_path, target_sr=16000):
    """
    Convert an MP3 file to WAV with a specific sample rate.

    Args:
        mp3_path: Path to the MP3 file
        target_sr: Target sample rate in Hz

    Returns:
        Path to the converted WAV file
    """
    try:
        # Load the MP3 file with librosa
        y, sr = librosa.load(mp3_path, sr=target_sr)

        # Create a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_wav.name, y, target_sr)
        temp_wav.close()

        return temp_wav.name
    except Exception as e:
        st.error(f"Error converting MP3 to WAV: {e}")
        return None

def generate_audio(text, lang='id', target_sr=16000):
    """
    Generate audio from text with a consistent sample rate.

    Args:
        text: The text to convert to speech
        lang: The language code (default: 'id' for Indonesian)
        target_sr: Target sample rate in Hz (default: 16000)

    Returns:
        Path to a temporary WAV file with the specified sample rate
    """
    # Use gTTS to generate speech
    tts = gTTS(text=text, lang=lang, slow=False)

    # Save to a temporary MP3 file first
    temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_mp3.name)
    temp_mp3.close()

    # Convert the MP3 to WAV with the target sample rate
    wav_path = convert_mp3_to_wav(temp_mp3.name, target_sr)

    # Clean up the MP3 file
    os.unlink(temp_mp3.name)

    return wav_path

def ensure_correct_sample_rate(audio_path, target_sr=16000):
    """
    Ensure audio file has the correct sample rate.

    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate in Hz

    Returns:
        Path to a WAV file with the specified sample rate
    """
    try:
        # Check if file is MP3
        if audio_path.lower().endswith('.mp3'):
            return convert_mp3_to_wav(audio_path, target_sr)

        # For WAV files, check the current sample rate
        y, sr = librosa.load(audio_path, sr=None)

        # If sample rate already matches, just return the path
        if sr == target_sr:
            return audio_path

        # Otherwise, resample and save to a temporary file
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Save to a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_wav.name, y_resampled, target_sr)
        temp_wav.close()

        return temp_wav.name
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        # Fall back to original file
        return audio_path

def normalize_audio(audio_path, target_sr=16000, target_level=-25):
    """
    Normalize audio to a target level and ensure consistent sample rate.

    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate in Hz
        target_level: Target RMS level in dB

    Returns:
        Path to normalized audio file
    """
    try:
        # Load the audio file with librosa
        y, sr = librosa.load(audio_path, sr=target_sr)

        # Calculate current RMS energy in dB
        rms = librosa.feature.rms(y=y)[0]
        rms_db = 20 * librosa.core.amplitude_to_db(rms.mean(), ref=1.0)

        # Calculate the gain needed
        gain_db = target_level - rms_db
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain to normalize volume
        y_normalized = y * gain_linear

        # Ensure we don't clip the audio
        if y_normalized.max() > 1.0:
            y_normalized = y_normalized / y_normalized.max() * 0.9  # Leave some headroom

        # Create a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_wav.name, y_normalized, target_sr)
        temp_wav.close()

        return temp_wav.name
    except Exception as e:
        st.error(f"Error normalizing audio: {e}")
        return audio_path  # Return original file as fallback

def try_download_with_path(bucket_name, file_path, target_sr):
    """
    Try to download an audio file with the given path.

    Args:
        bucket_name: GCS bucket name
        file_path: Path to the file in GCS
        target_sr: Target sample rate

    Returns:
        Audio file path if successful, None otherwise
    """
    try:
        # Ensure we have a valid GCS client
        if st.session_state.gcs_client is None:
            from app.data.gcs import initialize_gcs_client
            st.session_state.gcs_client = initialize_gcs_client(bucket_name)
            if st.session_state.gcs_client is None:
                return None

        # Download the audio file to a temporary location
        temp_audio_path = download_blob_to_temp(bucket_name, file_path)

        if temp_audio_path and os.path.exists(temp_audio_path):
            st.session_state.is_using_tts = False
            st.session_state.original_filename = file_path
            # Process the downloaded audio file to ensure correct format and sample rate
            return ensure_correct_sample_rate(temp_audio_path, target_sr)
        return None
    except Exception:
        # Silently fail and return None
        return None

def get_audio_from_gcs(sentence, sentences_df, bucket_name, fallback_to_tts=True, target_sr=16000):
    """
    Get audio for a sentence from Google Cloud Storage, handling extension mismatches silently.

    Args:
        sentence: The sentence text to find audio for
        sentences_df: DataFrame containing sentences and audio paths
        bucket_name: Google Cloud Storage bucket name
        fallback_to_tts: Whether to fallback to TTS if GCS file isn't found
        target_sr: Target sample rate in Hz

    Returns:
        Path to a temporary audio file with the specified sample rate and the original filename
    """
    if sentences_df is None or bucket_name is None:
        if fallback_to_tts:
            st.session_state.is_using_tts = True
            st.session_state.original_filename = None
            return generate_audio(sentence, target_sr=target_sr)
        return None

    # Look for the sentence in the DataFrame
    matches = sentences_df[sentences_df['sentence'] == sentence]

    if not matches.empty:
        # Use the first match
        file_path = matches.iloc[0]['path']
        original_path = file_path

        # Try the path as is first
        audio_path = try_download_with_path(bucket_name, file_path, target_sr)
        if audio_path:
            return audio_path

        # If that fails, try alternative extensions silently
        if file_path.lower().endswith('.wav'):
            # Try with .mp3 instead
            mp3_path = file_path[:-4] + '.mp3'
            audio_path = try_download_with_path(bucket_name, mp3_path, target_sr)
            if audio_path:
                # Update the path in the DataFrame for future reference
                if 'path' in sentences_df.columns:
                    idx = sentences_df[sentences_df['path'] == original_path].index
                    if len(idx) > 0:
                        sentences_df.loc[idx, 'path'] = mp3_path
                return audio_path
        elif file_path.lower().endswith('.mp3'):
            # Try with .wav instead
            wav_path = file_path[:-4] + '.wav'
            audio_path = try_download_with_path(bucket_name, wav_path, target_sr)
            if audio_path:
                # Update the path in the DataFrame for future reference
                if 'path' in sentences_df.columns:
                    idx = sentences_df[sentences_df['path'] == original_path].index
                    if len(idx) > 0:
                        sentences_df.loc[idx, 'path'] = wav_path
                return audio_path

        # If we're here, file wasn't found with either extension
        if fallback_to_tts:
            st.session_state.is_using_tts = True
            st.session_state.original_filename = original_path
            return generate_audio(sentence, target_sr=target_sr)
        return None
    else:
        # No matching sentence found
        if fallback_to_tts:
            st.session_state.is_using_tts = True
            st.session_state.original_filename = None
            return generate_audio(sentence, target_sr=target_sr)
        return None

def get_audio_for_sentence(sentence, common_voice_data, bucket_name, fallback_to_tts=True, target_sr=16000):
    """
    Get audio for a sentence from Google Cloud Storage.

    Args:
        sentence: The sentence text to find audio for
        common_voice_data: The Common Voice dataset (not used for GCS)
        bucket_name: Google Cloud Storage bucket name
        fallback_to_tts: Whether to fallback to TTS if GCS file isn't found
        target_sr: Target sample rate in Hz

    Returns:
        Path to an audio file with the specified sample rate
    """
    # Check if we have a DataFrame of sentences
    if st.session_state.sentences_df is not None:
        return get_audio_from_gcs(sentence, st.session_state.sentences_df, bucket_name, fallback_to_tts, target_sr)

    # If no DataFrame is available, we can't locate audio files in GCS
    # So we'll use TTS as a fallback
    if fallback_to_tts:
        st.session_state.is_using_tts = True
        st.session_state.original_filename = None
        return generate_audio(sentence, target_sr=target_sr)
    else:
        st.error(f"No reference audio found for: '{sentence}' (no sentence database available)")
        return None
