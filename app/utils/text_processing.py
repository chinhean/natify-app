"""
Text processing utilities for the Indonesian Pronunciation App.
"""

import re
import string
from difflib import SequenceMatcher

def normalize_text(text):
    """
    Normalize text by converting to lowercase, removing punctuation, and extra whitespace.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def compare_text_content(expected_text, recognized_text):
    """
    Compare the recognized text with the expected sentence using sequence matching.
    This ensures consistency between the content score and feedback.

    Args:
        expected_text: The expected sentence text
        recognized_text: The recognized text from speech recognition

    Returns:
        Content similarity score (0-100)
    """
    if not recognized_text:
        return 0  # If no speech was recognized, return a score of 0

    # Clean and normalize both texts
    normalized_expected = normalize_text(expected_text)
    normalized_recognized = normalize_text(recognized_text)

    # If exact match after normalization, return perfect score
    if normalized_expected == normalized_recognized:
        return 100

    # Calculate similarity using SequenceMatcher
    sequence_similarity = SequenceMatcher(None, normalized_expected, normalized_recognized).ratio() * 100

    # Calculate word-level similarity for a more balanced score
    expected_words = normalized_expected.split()
    recognized_words = normalized_recognized.split()

    word_matches = 0
    for ew in expected_words:
        min_distance = float('inf')
        best_match = None

        for rw in recognized_words:
            # Use sequence matcher for word comparison
            ratio = SequenceMatcher(None, ew, rw).ratio()
            distance = 1 - ratio  # Convert similarity to distance

            if distance < min_distance:
                min_distance = distance
                best_match = rw

        # If there's a close match (similarity > 80%)
        if best_match and min_distance < 0.2:  # 80% similarity threshold
            word_matches += 1

    word_score = (word_matches / len(expected_words)) * 100 if expected_words else 0

    # Weighted combination with more emphasis on sequence similarity
    content_score = (sequence_similarity * 0.7) + (word_score * 0.3)

    return content_score
