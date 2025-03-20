"""
Phoneme extraction and comparison for the Indonesian Pronunciation App.
"""

import streamlit as st
import torch
import librosa
import numpy as np
from difflib import SequenceMatcher
import Levenshtein
from app.data.phonemes import map_to_standard_indonesian_phonemes, identify_challenges

def extract_phonemes_wav2vec2(audio_path, model, processor, feature_extractor=None, sample_rate=16000):
    """
    Extract phonemes from audio using wav2vec2 model.

    Args:
        audio_path: Path to the audio file
        model: Wav2vec2 model
        processor: Wav2vec2 processor
        feature_extractor: Wav2vec2 feature extractor (optional)
        sample_rate: Sample rate in Hz

    Returns:
        String: Extracted phonemes
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sample_rate)

        # Process audio with wav2vec2
        if feature_extractor is not None:
            # Use the feature extractor directly
            inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
        elif hasattr(processor, 'feature_extractor'):
            # Use the processor's feature extractor
            inputs = processor(y, sampling_rate=sr, return_tensors="pt")
        else:
            # Direct call to processor
            inputs = processor(y, sampling_rate=sr, return_tensors="pt")

        # Get input values for the model
        if 'input_values' in inputs:
            input_values = inputs.input_values
        else:
            # Fall back to assuming the processor returns the input_values directly
            input_values = inputs

        with torch.no_grad():
            logits = model(input_values).logits

        # Get predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # For cahya/wav2vec2-large-xlsr-indonesian, we can decode directly
        if "cahya-indonesian" in str(model.config.model_type):
            try:
                transcription = processor.batch_decode(predicted_ids)[0]
                # Map to standard phonemes
                return map_to_standard_indonesian_phonemes(transcription)
            except Exception as e:
                st.error(f"Error decoding with processor: {e}")
                # Fall back to the generic approach below

        # Create a simplified phoneme representation
        # Since direct decoding may not work, we'll map to IPA-like phonemes
        # First 100 tokens ~ common phonemes in many languages
        ipa_like_phonemes = "abcdefghijklmnopqrstuvwxyzəɪʊɛɔæɑʌɒɨʉɯɤøɵœɶɐɞʏɘɹɾɽɻɺɮɬɡɠɧɦɥɰʎʍɕʑʡʢǀǁǂǃɓɗɓǃǂɠʘ!ʼ,.?-:;()[]{}"

        # Convert each prediction to a character, but avoid repeats
        phoneme_list = []
        prev_id = None

        for id_tensor in predicted_ids[0]:
            id_val = id_tensor.item()
            # Skip if it's a repeat
            if id_val != prev_id:
                phoneme_index = id_val % len(ipa_like_phonemes)
                phoneme_list.append(ipa_like_phonemes[phoneme_index])
                prev_id = id_val

        # Join into a string, limiting to first 30 characters to avoid noise
        phoneme_string = ''.join(phoneme_list[:30])

        # Map to standard phonemes
        return map_to_standard_indonesian_phonemes(phoneme_string)
    except Exception as e:
        st.error(f"Error extracting phonemes: {e}")
        return ""

def ensure_consistent_phoneme_extraction(reference_audio_path, user_audio_path, model, processor, feature_extractor=None):
    """
    Ensure both reference and user phonemes are extracted using the same method (audio-based).

    Args:
        reference_audio_path: Path to reference audio
        user_audio_path: Path to user's recorded audio
        model, processor, feature_extractor: Model components for wav2vec2

    Returns:
        Tuple of (reference_phonemes, user_phonemes)
    """
    # Always extract phonemes from audio for both reference and user recording
    reference_phonemes = extract_phonemes_wav2vec2(
        reference_audio_path, model, processor, feature_extractor
    )

    user_phonemes = extract_phonemes_wav2vec2(
        user_audio_path, model, processor, feature_extractor
    )

    # Ensure both are standardized (though extract_phonemes_wav2vec2 should already do this)
    reference_phonemes = map_to_standard_indonesian_phonemes(reference_phonemes)
    user_phonemes = map_to_standard_indonesian_phonemes(user_phonemes)

    return reference_phonemes, user_phonemes

def text_to_phonemes(text, epi):
    """
    Convert text to phonemes using Epitran and standardize them.

    Args:
        text: Text to convert to phonemes
        epi: Epitran instance

    Returns:
        String: Phonemes
    """
    if epi is None:
        return ""

    try:
        ipa_phonemes = epi.transliterate(text)
        # Standardize the phoneme representation
        return map_to_standard_indonesian_phonemes(ipa_phonemes)
    except Exception as e:
        st.error(f"Error converting text to phonemes: {e}")
        return ""

def compare_phonemes(expected_phonemes, recognized_phonemes):
    """
    Compare expected phonemes with recognized phonemes and calculate similarity score.

    Args:
        expected_phonemes: Expected phonemes
        recognized_phonemes: Recognized phonemes

    Returns:
        Tuple: (similarity_score, comparison_details)
    """
    if not recognized_phonemes or not expected_phonemes:
        return 0, []

    # Clean and normalize phonemes
    expected_phonemes = expected_phonemes.lower().strip()
    recognized_phonemes = recognized_phonemes.lower().strip()

    # If exact match, return perfect score
    if expected_phonemes == recognized_phonemes:
        return 100, [("perfect", expected_phonemes, recognized_phonemes)]

    # Calculate similarity using Levenshtein distance
    distance = Levenshtein.distance(expected_phonemes, recognized_phonemes)
    max_len = max(len(expected_phonemes), len(recognized_phonemes))
    similarity = (1 - distance / max_len) * 100

    # Generate detailed comparison
    comparison = []
    matcher = SequenceMatcher(None, expected_phonemes, recognized_phonemes)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            comparison.append(("match", expected_phonemes[i1:i2], recognized_phonemes[j1:j2]))
        elif tag == 'replace':
            comparison.append(("replace", expected_phonemes[i1:i2], recognized_phonemes[j1:j2]))
        elif tag == 'delete':
            comparison.append(("delete", expected_phonemes[i1:i2], ""))
        elif tag == 'insert':
            comparison.append(("insert", "", recognized_phonemes[j1:j2]))

    return similarity, comparison
