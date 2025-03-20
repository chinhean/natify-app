"""
Machine learning model loading and management for the Indonesian Pronunciation App.
"""

import streamlit as st
import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    MarianMTModel,
    MarianTokenizer
)
import epitran
import whisper

@st.cache_resource
def load_wav2vec2_model(model_name):
    """
    Load the wav2vec2 model for phoneme recognition with improved model-specific handling.

    Args:
        model_name: Name or path of the model to load

    Returns:
        Tuple: (model, processor, feature_extractor)
    """
    try:
        # For cahya/wav2vec2-large-xlsr-indonesian model
        if "cahya/wav2vec2-large-xlsr-indonesian" in model_name:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)

            # Store model-specific configuration for extraction
            model.config.model_type = "cahya-indonesian"
            return model, processor, None

        # For wav2vec2-lv-60-espeak-cv-ft, we need to handle differently
        elif "wav2vec2-lv-60-espeak-cv-ft" in model_name:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)

            # Store model-specific configuration
            model.config.model_type = "wav2vec2-phoneme"

            # Create a custom processor with improved handling
            class EnhancedProcessor:
                def __init__(self, feature_extractor, model_name):
                    self.feature_extractor = feature_extractor
                    self.model_name = model_name

                def __call__(self, *args, **kwargs):
                    return self.feature_extractor(*args, **kwargs)

                # Try to implement decode method even for feature extractors
                def batch_decode(self, token_ids):
                    # This model uses IPA phonemes via espeak
                    # We can map token IDs to phonemes as best as possible
                    try:
                        # For this model specifically, we use a more stable mapping
                        # This is a placeholder for a proper phoneme mapping
                        return [" ".join([str(token_id.item()) for token_id in ids])
                                for ids in token_ids]
                    except Exception as e:
                        return [f"Error decoding: {e}"]

            processor = EnhancedProcessor(feature_extractor, model_name)
            return model, processor, feature_extractor

        # For the xlsr-53 model
        elif "wav2vec2-large-xlsr-53" in model_name:
            # Try using AutoProcessor which can be more flexible
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(model_name)
                model = Wav2Vec2ForCTC.from_pretrained(model_name)

                # Store model-specific configuration
                model.config.model_type = "xlsr-53"

                return model, processor, None
            except:
                # Fallback to feature extractor only
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                model = Wav2Vec2ForCTC.from_pretrained(model_name)

                # Store model-specific configuration
                model.config.model_type = "xlsr-53"

                # Create an enhanced processor
                class EnhancedProcessor:
                    def __init__(self, feature_extractor, model_name):
                        self.feature_extractor = feature_extractor
                        self.model_name = model_name

                    def __call__(self, *args, **kwargs):
                        return self.feature_extractor(*args, **kwargs)

                    # Add batch_decode method for consistency
                    def batch_decode(self, token_ids):
                        # This is a placeholder mapping
                        return [" ".join([str(token_id.item()) for token_id in ids])
                                for ids in token_ids]

                processor = EnhancedProcessor(feature_extractor, model_name)
                return model, processor, feature_extractor

        # Default approach for other models
        else:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)

            # Store model-specific configuration
            model.config.model_type = "default"

            return model, processor, None

    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None, None, None

@st.cache_resource
def load_whisper_model(model_size="base"):
    """
    Load the OpenAI Whisper model for speech recognition.

    Args:
        model_size: Size of the Whisper model (tiny, base, small, medium, large)

    Returns:
        Loaded Whisper model
    """
    try:
        model = whisper.load_model(model_size)
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

@st.cache_resource
def load_translation_model():
    """
    Load the MarianMT model for Indonesian to English translation.

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        model_name = "Helsinki-NLP/opus-mt-id-en"  # Indonesian to English
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading translation model: {e}")
        return None, None

@st.cache_resource
def load_epitran():
    """
    Load Epitran for text-to-phoneme conversion.

    Returns:
        Epitran instance
    """
    try:
        return epitran.Epitran('ind-Latn')
    except Exception as e:
        st.error(f"Error loading Epitran: {e}")
        return None
