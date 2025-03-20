"""
Phoneme data and mappings for the Indonesian Pronunciation App.
"""

# Define phoneme mapping for Indonesian
indonesian_phonemes = {
    'a': 'a',
    'b': 'b',
    'c': 'tʃ',
    'd': 'd',
    'e': 'ə',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'i': 'i',
    'j': 'dʒ',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'o',
    'p': 'p',
    'q': 'k',
    'r': 'r',
    's': 's',
    't': 't',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'x': 'ks',
    'y': 'j',
    'z': 'z',
    'ng': 'ŋ',
    'ny': 'ɲ',
    'sy': 'ʃ',
    'kh': 'x',
}

def map_to_standard_indonesian_phonemes(phoneme_string):
    """
    Map the extracted phoneme symbols to standard Indonesian phoneme representations

    Args:
        phoneme_string: Raw phoneme string from the extraction model

    Returns:
        String with standardized Indonesian phoneme representations
    """
    # Define mappings from raw extracted symbols to standard IPA for Indonesian
    indonesian_phoneme_map = {
        # Vowels
        'a': 'a',   # like in "father"
        'i': 'i',   # like in "machine"
        'u': 'u',   # like in "food"
        'e': 'e',   # like in "bet"
        'ə': 'ə',   # schwa sound
        'o': 'o',   # like in "go"

        # Consonants
        'b': 'b',
        'c': 'tʃ',  # like "ch" in "chair"
        'd': 'd',
        'f': 'f',
        'g': 'g',
        'h': 'h',
        'j': 'dʒ',  # like "j" in "jump"
        'k': 'k',
        'l': 'l',
        'm': 'm',
        'n': 'n',
        'p': 'p',
        'q': 'k',
        'r': 'r',   # slightly rolled
        's': 's',
        't': 't',
        'v': 'v',
        'w': 'w',
        'x': 'ks',
        'y': 'j',
        'z': 'z',

        # Special consonant clusters
        'ng': 'ŋ',  # like in "singer"
        'ny': 'ɲ',  # like "ny" in "canyon"
        'sy': 'ʃ',  # like "sh" in "ship"
        'kh': 'x',  # velar fricative
    }

    # Process the string to map to standard representations
    standardized = ""
    i = 0
    while i < len(phoneme_string):
        # Check for digraphs first (2-character phonemes)
        if i < len(phoneme_string) - 1:
            digraph = phoneme_string[i:i+2]
            if digraph in indonesian_phoneme_map:
                standardized += indonesian_phoneme_map[digraph]
                i += 2
                continue

        # Handle single characters
        char = phoneme_string[i]
        if char in indonesian_phoneme_map:
            standardized += indonesian_phoneme_map[char]
        else:
            # If not in map, keep original
            standardized += char
        i += 1

    return standardized

# Common pronunciation challenges for Indonesian learners
pronunciation_challenges = {
    'r': "The Indonesian 'r' is slightly rolled or trilled, similar to Spanish 'r'",
    'c': "The Indonesian 'c' is pronounced like 'ch' in 'chair', not like 'k' or 's'",
    'e': "Indonesian has two 'e' sounds: 'e' as in 'bet' and 'e' as a schwa (ə) like in 'the'",
    'ng': "The 'ng' sound is pronounced as a single sound like in 'singer', not 'n'+'g'",
    'ny': "The 'ny' sound is pronounced as a single sound like 'ñ' in Spanish or 'gn' in Italian",
    'j': "The 'j' is pronounced like 'j' in 'jump', not like 'y' in 'yes'",
    'u': "The 'u' is pronounced like 'oo' in 'food', not like 'u' in 'but'",
    'ai': "The diphthong 'ai' is pronounced like 'eye', not as separate vowels",
    'au': "The diphthong 'au' is pronounced like 'ow' in 'how', not as separate vowels"
}

def identify_challenges(phoneme_comparison, pronunciation_challenges):
    """
    Identify common pronunciation challenges based on phoneme comparison.

    Args:
        phoneme_comparison: List of tuples with phoneme comparison details
        pronunciation_challenges: Dictionary of common pronunciation challenges

    Returns:
        List of tuples with (challenge, description)
    """
    challenges = []

    for match_type, expected, actual in phoneme_comparison:
        if match_type != "match":
            # Check for specific challenging phonemes
            for key, description in pronunciation_challenges.items():
                if key in expected:
                    if (match_type == "replace" and key not in actual) or match_type == "delete":
                        challenges.append((key, description))

    # Remove duplicates while preserving order
    return list(dict.fromkeys(challenges))
