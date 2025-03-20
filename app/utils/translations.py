"""
Translation utilities for the Indonesian Pronunciation App.
"""

import streamlit as st
from app.ml_logic.models import load_translation_model

def translate_text(text, max_length=50):
    """
    Translate Indonesian text to English using MarianMT.

    Args:
        text: Text to translate
        max_length: Maximum length of text chunk for processing

    Returns:
        Translated text
    """
    model, tokenizer = load_translation_model()
    if model is None or tokenizer is None:
        return "Translation not available"

    try:
        # Split long text to avoid issues
        if len(text.split()) > max_length:
            parts = []
            words = text.split()
            for i in range(0, len(words), max_length):
                part = " ".join(words[i:i+max_length])
                inputs = tokenizer([part], return_tensors="pt", padding=True)
                outputs = model.generate(**inputs)
                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                parts.append(translation)
            return " ".join(parts)
        else:
            inputs = tokenizer([text], return_tensors="pt", padding=True)
            outputs = model.generate(**inputs)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
    except Exception as e:
        st.warning(f"Error in translation: {e}")
        return "Translation error"
