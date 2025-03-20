"""
Feedback components for the Indonesian Pronunciation App.
"""

import streamlit as st
import librosa
import re
from app.data.phonemes import pronunciation_challenges

def display_pronunciation_feedback(user_recording, reference_recording, phoneme_comparison, recognized_text, expected_text):
    """
    Display learner-friendly pronunciation feedback with clearer organization and more encouraging language.
    Fixed to ensure content accuracy percentage matches the main content score.

    Args:
        user_recording: Path to user's recording
        reference_recording: Path to reference recording
        phoneme_comparison: List of tuples with phoneme comparison details
        recognized_text: Text recognized from user's recording
        expected_text: Expected text/sentence
    """
    st.markdown("## 🎤 Pronunciation Feedback")
    st.markdown("---")

    # Extract recording durations for rhythm comparison
    try:
        user_y, sr = librosa.load(user_recording, sr=16000)
        ref_y, _ = librosa.load(reference_recording, sr=16000)

        user_duration = librosa.get_duration(y=user_y, sr=sr)
        ref_duration = librosa.get_duration(y=ref_y, sr=sr)

        duration_ratio = min(user_duration, ref_duration) / max(user_duration, ref_duration)

        # Speaking pace feedback
        st.markdown("### 🏃‍♂️ Speaking Pace")
        if duration_ratio < 0.7:
            if user_duration > ref_duration:
                st.info("Your speaking is a bit slower than the native example. That's perfectly fine for learning! As you become more comfortable, you can gradually increase your pace.")
            else:
                st.info("You're speaking a bit faster than the native example. It's great you're confident! Try slowing down slightly to focus on each sound.")
        else:
            st.success("Your speaking pace matches the native example very well! This is excellent for developing natural-sounding Indonesian.")
    except Exception as e:
        pass  # Silently handle errors

    # Content accuracy feedback with consistent calculation
    if recognized_text and expected_text:
        st.markdown("### 📝 Content Accuracy")

        # Simply use the same content_score from the session state
        # This ensures the feedback exactly matches the score shown in the summary
        if 'content_score' in st.session_state and st.session_state.content_score is not None:
            content_accuracy = st.session_state.content_score
        else:
            # Fallback calculation if not in session state (shouldn't normally happen)
            from app.utils.text_processing import compare_text_content
            content_accuracy = compare_text_content(expected_text, recognized_text)

        # Normalize texts for word identification only
        def normalize_text(text):
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text

        normalized_expected = normalize_text(expected_text)
        normalized_recognized = normalize_text(recognized_text)

        # Identify missing words for helpful feedback
        expected_words = normalized_expected.split()
        recognized_words = normalized_recognized.split()

        # Find words to focus on
        missing_words = set()

        for exp_word in expected_words:
            if exp_word not in recognized_words:
                missing_words.add(exp_word)

        # Display feedback message based on the EXACT same score shown in summary
        if content_accuracy >= 80:
            st.success(f"Great job! Your content accuracy score is {content_accuracy:.1f}%.")
        elif content_accuracy >= 60:
            st.info(f"Good effort! Your content accuracy score is {content_accuracy:.1f}%. Keep practicing.")
        else:
            st.info(f"Your content accuracy score is {content_accuracy:.1f}%. Focus on the words below to improve.")

        # Hide the word-level accuracy since it's confusing
        # but keep the "Words to Focus On" section which is valuable
        if missing_words:
            st.markdown("#### Words to Focus On:")
            st.markdown("These words might need a bit more practice:")

            for word in missing_words:
                st.markdown(f"• **{word}**")

            # Keep the helpful tips for pronunciations
            st.markdown("#### Helpful Tips:")
            provided_tips = set()
            tips_found = False

            for word in missing_words:
                for phoneme, description in pronunciation_challenges.items():
                    if phoneme in word and phoneme not in provided_tips:
                        st.info(f"For '{phoneme}' in '{word}': {description}")
                        provided_tips.add(phoneme)
                        tips_found = True
                        if len(provided_tips) >= 3:
                            break

            if not tips_found:
                st.markdown("Try listening to the native audio again and focus on mimicking the sounds closely.")
        else:
            st.success("All words were recognized correctly. Great job!")

    # Phoneme-level feedback
    if phoneme_comparison:
        st.markdown("### 🔊 Sound Accuracy")

        # Count different types of phoneme issues
        match_count = 0
        replace_count = 0
        delete_count = 0
        insert_count = 0

        # Track which specific sounds were problematic for each type of issue
        problem_phonemes = {}
        replaced_sounds = []
        deleted_sounds = []
        inserted_sounds = []

        for match_type, expected, actual in phoneme_comparison:
            if match_type == "match" or match_type == "perfect":
                match_count += 1
            elif match_type == "replace":
                replace_count += 1
                replaced_sounds.append((expected, actual))
                # Track the problematic phoneme with more detail
                if expected:
                    if expected in problem_phonemes:
                        problem_phonemes[expected] = problem_phonemes[expected] + [f"replaced with '{actual}'"]
                    else:
                        problem_phonemes[expected] = [f"replaced with '{actual}'"]
            elif match_type == "delete":
                delete_count += 1
                deleted_sounds.append(expected)
                # Track the missing phoneme
                if expected:
                    if expected in problem_phonemes:
                        problem_phonemes[expected] = problem_phonemes[expected] + ["missing"]
                    else:
                        problem_phonemes[expected] = ["missing"]
            elif match_type == "insert":
                insert_count += 1
                inserted_sounds.append(actual)

        total_phonemes = match_count + replace_count + delete_count
        if total_phonemes > 0:
            accuracy = (match_count / total_phonemes) * 100
        else:
            accuracy = 0

        # Use phoneme_score from session state if available (for consistency)
        if 'phoneme_score' in st.session_state and st.session_state.phoneme_score is not None:
            accuracy = st.session_state.phoneme_score

        # Overall phoneme feedback with more encouraging language
        if accuracy >= 80:
            st.success(f"Excellent sound pronunciation! You matched {accuracy:.1f}% of the sounds perfectly.")
        elif accuracy >= 60:
            st.warning(f"Good work! You matched {accuracy:.1f}% of the sounds. With practice, you'll improve even more.")
        else:
            st.info(f"You matched {accuracy:.1f}% of the sounds. Indonesian has some unique sounds that take time to master - keep practicing!")

        # Make sound patterns much more specific
        if replace_count > 0 or delete_count > 0 or insert_count > 0:
            st.markdown("#### Sound Patterns:")

            # Show specifically which sounds were replaced
            if replace_count > 0:
                replacements = []
                for expected, actual in replaced_sounds:
                    replacements.append(f"'{expected}' → '{actual}'")

                st.markdown("**Sounds you pronounced differently:**")
                for i, replacement in enumerate(replacements[:3]):  # Limit to first 3
                    st.markdown(f"• {replacement}")
                if len(replacements) > 3:
                    st.markdown(f"• ... and {len(replacements) - 3} more")

            # Show specifically which sounds were missing
            if delete_count > 0:
                st.markdown("**Sounds that were missing:**")
                for i, sound in enumerate(deleted_sounds[:3]):  # Limit to first 3
                    st.markdown(f"• '{sound}'")
                if len(deleted_sounds) > 3:
                    st.markdown(f"• ... and {len(deleted_sounds) - 3} more")

            # Show specifically which extra sounds were added
            if insert_count > 0:
                st.markdown("**Extra sounds you added:**")
                for i, sound in enumerate(inserted_sounds[:3]):  # Limit to first 3
                    st.markdown(f"• '{sound}'")
                if len(inserted_sounds) > 3:
                    st.markdown(f"• ... and {len(inserted_sounds) - 3} more")

        # Show specific sounds to practice with actual examples from the sentence
        if problem_phonemes:
            st.markdown("#### Specific Sounds to Practice:")

            # Sort problems by frequency
            sorted_problems = sorted(problem_phonemes.items(),
                                    key=lambda x: len(x[1]),
                                    reverse=True)

            # Find examples of each problematic sound in the original sentence
            def find_example_words(phoneme, sentence):
                example_words = []
                for word in sentence.lower().split():
                    if phoneme.lower() in word:
                        example_words.append(word)
                return example_words[:2]  # Return up to 2 examples

            # Display each problematic sound with examples
            for phoneme, issues in sorted_problems[:3]:  # Limit to top 3 issues
                # Get the description from pronunciation_challenges
                description = pronunciation_challenges.get(phoneme, "Focus on this sound")

                # Find examples in the original sentence
                examples = find_example_words(phoneme, expected_text)
                example_text = ""
                if examples:
                    example_text = f"Examples in this sentence: {', '.join(['**'+word+'**' for word in examples])}"

                # Format the issue text
                issue_text = ", ".join(issues[:2])  # Limit to first 2 issues per sound
                if len(issues) > 2:
                    issue_text += ", etc."

                # Create the full feedback with sound, description, issue and examples
                st.info(f"**Sound '{phoneme}'**:  \n{description}  \n*Issue: {issue_text}*  \n{example_text}")
        else:
            st.success("No specific sound issues identified. Keep practicing for even better pronunciation!")

    # Practice recommendations with clearer section
    st.markdown("### 🔄 Practice Strategy")
    st.markdown("""
    1. **Listen** to the reference audio multiple times
    2. **Focus** on the specific sounds mentioned above
    3. **Record** yourself speaking slowly and clearly
    4. **Compare** your recording with the reference
    5. **Repeat** daily for best results

    Remember: Learning pronunciation takes time. Each practice session brings you closer to fluency!
    """)
