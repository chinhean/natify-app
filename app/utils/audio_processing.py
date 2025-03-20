"""
Audio processing utilities for the Indonesian Pronunciation App.
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
import tempfile
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def extract_features(audio_path, sample_rate=16000):
    """
    Extract multiple types of audio features for more robust comparison.

    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate

    Returns:
        Dictionary of feature arrays
    """
    y, sr = librosa.load(audio_path, sr=sample_rate)

    # Extract multiple feature types
    features = {}

    # 1. MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 2. Delta MFCCs (first order derivatives)
    mfcc_delta = librosa.feature.delta(mfccs)

    # 3. Delta-delta MFCCs (second order derivatives)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    # 4. Spectral centroid - captures the "brightness" of the sound
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # 5. Spectral bandwidth - captures the width of the spectrum
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # 6. Spectral rolloff - captures the frequency below which most energy is contained
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # 7. Zero crossing rate - related to noisiness and consonant sounds
    zcr = librosa.feature.zero_crossing_rate(y)

    # Store all features
    features['mfccs'] = mfccs.T  # Transpose for time dimension first
    features['mfcc_delta'] = mfcc_delta.T
    features['mfcc_delta2'] = mfcc_delta2.T
    features['spectral_centroid'] = centroid.T
    features['spectral_bandwidth'] = bandwidth.T
    features['spectral_rolloff'] = rolloff.T
    features['zcr'] = zcr.T

    return features

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
        rms_db = 20 * np.log10(np.mean(rms) + 1e-8)  # Add small constant to avoid log(0)

        # Calculate the gain needed
        gain_db = target_level - rms_db
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain to normalize volume
        y_normalized = y * gain_linear

        # Ensure we don't clip the audio
        if np.max(np.abs(y_normalized)) > 1.0:
            y_normalized = y_normalized / np.max(np.abs(y_normalized)) * 0.9  # Leave some headroom

        # Create a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_wav.name, y_normalized, target_sr)
        temp_wav.close()

        return temp_wav.name
    except Exception as e:
        st.error(f"Error normalizing audio: {e}")
        return audio_path  # Return original file as fallback

def compare_acoustic_features(reference_path, user_path, recognized_phonemes=None):
    """
    Compare acoustic features using multiple feature types with balanced, realistic scoring.
    Provides more accurate assessment of pronunciation quality with appropriate penalties
    and less artificial boosting.

    Args:
        reference_path: Path to reference audio file
        user_path: Path to user's recorded audio file
        recognized_phonemes: Phonemes recognized in user's recording (if None, will proceed with acoustic analysis)

    Returns:
        Final acoustic score (0-100)
    """
    # If no phonemes were recognized, immediately return 0
    if recognized_phonemes is not None and (recognized_phonemes == "" or recognized_phonemes is None):
        st.warning("No speech detected by phoneme recognition. Please speak clearly.")
        return 0

    try:
        # First normalize both audio files to ensure consistent volume levels
        norm_reference_path = normalize_audio(reference_path)
        norm_user_path = normalize_audio(user_path)

        # Extract enhanced features from both audio files
        reference_features = extract_features(norm_reference_path)
        user_features = extract_features(norm_user_path)

        # More balanced feature weights - reduce MFCC dominance
        feature_weights = {
            'mfccs': 0.35,           # MFCCs are most important for pronunciation
            'mfcc_delta': 0.15,      # Delta features capture transitions
            'mfcc_delta2': 0.1,      # Delta-delta features capture acceleration
            'spectral_centroid': 0.1,
            'spectral_bandwidth': 0.1,
            'spectral_rolloff': 0.1,
            'zcr': 0.1
        }

        # Calculate length penalty with realistic strictness
        ref_duration = librosa.get_duration(path=norm_reference_path)
        user_duration = librosa.get_duration(path=norm_user_path)

        duration_ratio = min(ref_duration, user_duration) / max(ref_duration, user_duration)

        # Apply a more appropriate non-linear penalty (0.3 instead of 0.15)
        length_score = 100 * (duration_ratio ** 0.3)

        # Apply a significant penalty for very short recordings
        if duration_ratio < 0.5:
            length_score *= 0.75  # 25% penalty instead of just 3%

        # Calculate distance-based scores for each feature type
        feature_scores = {}

        for feature_name, weight in feature_weights.items():
            ref_feat = reference_features[feature_name]
            user_feat = user_features[feature_name]

            # Handle different lengths in features
            min_length = min(len(ref_feat), len(user_feat))

            # If too different in length, apply a meaningful penalty
            if abs(len(ref_feat) - len(user_feat)) / len(ref_feat) > 0.3:
                feature_scores[feature_name] = 60  # Lower base score (was 80)
            else:
                # Use DTW for temporal alignment with balanced feature emphasis
                if feature_name in ['mfccs', 'mfcc_delta', 'mfcc_delta2']:
                    # For MFCC features, use DTW with full alignment
                    distance, path = fastdtw(ref_feat[:min_length], user_feat[:min_length],
                                           dist=euclidean)

                    # Normalize by path length and feature dimensionality
                    avg_dist = distance / (len(path) * ref_feat.shape[1])

                    # Apply more balanced sigmoid-like scaling (2.0 instead of 1.8)
                    score = 100 / (1 + np.exp(avg_dist - 2.0))

                    # Apply a moderate boost to MFCC scores
                    score = min(100, score * 1.05)  # 5% boost with 100% cap (was 18%)
                else:
                    # For other spectral features, use simpler comparison
                    # Compare overall distributions rather than exact alignment
                    ref_mean = np.mean(ref_feat)
                    user_mean = np.mean(user_feat)

                    # Calculate relative difference
                    rel_diff = abs(ref_mean - user_mean) / (ref_mean + 1e-8)

                    # Apply more balanced non-linear scaling (1.0 instead of 0.8)
                    score = 100 * np.exp(-1.0 * rel_diff)

                    # Apply modest boost to spectral feature scores
                    score = min(100, score * 1.05)  # 5% boost with 100% cap (was 12%)

                feature_scores[feature_name] = score

        # Combine scores with weights, with more balanced importance
        weighted_score = sum(score * feature_weights[name]
                           for name, score in feature_scores.items())

        # Apply length penalty with appropriate weight
        final_score = (weighted_score * 0.8) + (length_score * 0.2)  # 0.8/0.2 instead of 0.9/0.1

        # Add a modest progressive boost for good matches
        if weighted_score > 80:  # Higher threshold (was 75)
            # Calculate a moderate boost
            boost_factor = 1.0 + (weighted_score - 80) * 0.002  # Lower factor 0.002 (was 0.006)
            final_score = final_score * boost_factor

        # Add a modest fixed boost for exceptional matches with stricter conditions
        if weighted_score > 90 and length_score > 85:  # Stricter conditions (was 85/80)
            final_score = final_score * 1.05  # 5% boost (was 12%)

        # Ensure the score is in the range [0, 100]
        final_score = max(0, min(100, final_score))

        # Clean up temporary files
        try:
            os.unlink(norm_reference_path)
            os.unlink(norm_user_path)
        except:
            pass

        return final_score

    except Exception as e:
        st.error(f"Error comparing acoustic features: {e}")
        return 0  # Return 0 on error
