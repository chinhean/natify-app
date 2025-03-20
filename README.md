# Natify: Indonesian Pronunciation Coach

Natify is a web application that helps users learn and improve their Indonesian pronunciation through AI-powered feedback and analysis.

## Features

- **Audio Recording and Playback**: Record your pronunciation attempts and compare them with native speakers
- **Multilevel Phoneme Analysis**: Get detailed feedback on your pronunciation at the phoneme level
- **Speech Recognition**: Verify if what you said matches the expected sentence
- **Visual Feedback**: See visualizations of your pronunciation compared to the reference
- **Difficulty Levels**: Practice with sentences of varying complexity (easy, medium, difficult)
- **Progress Tracking**: Track your improvement over time
- **Google Cloud Storage Integration**: Use your own custom sentence collections

## Technology Stack

- **Streamlit**: For the web interface
- **wav2vec2**: For phoneme recognition
- **Whisper/Google STT**: For speech-to-text conversion
- **LibROSA/SoundDevice**: For audio processing and analysis
- **Google Cloud Storage**: For storing and retrieving sentence data and audio files
- **MarianMT**: For machine translation

## Installation

### Prerequisites

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) for audio processing
- [PortAudio](http://www.portaudio.com/download.html) for audio recording
- Google Cloud credentials (if using GCS features)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/natify.git
   cd natify
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure Google Cloud Storage:
   - Create a Google Cloud project and enable the Storage API
   - Create a service account with Storage Object Viewer role
   - Download the service account key file
   - Set the environment variable:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"
     ```

## Running the App

```bash
streamlit run app/main.py
```

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t natify .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 -v /path/to/your-key.json:/app/gcp-credentials.json natify
   ```

3. Access the app at http://localhost:8501

## Cloud Deployment

### Google Cloud Run

1. Build and push the Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/your-project-id/natify
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy natify \
     --image gcr.io/your-project-id/natify \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json" \
     --set-secrets "gcp-credentials.json=your-secret:latest"
   ```

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Configure secrets for Google Cloud credentials

## Project Structure

```
natify/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Main Streamlit app entry point
│   ├── interface/              # UI components
│   │   ├── __init__.py
│   │   ├── sidebar.py          # Sidebar components
│   │   ├── feedback.py         # Pronunciation feedback UI
│   │   └── audio.py            # Audio recording/playback components
│   ├── ml_logic/               # ML model handling
│   │   ├── __init__.py
│   │   ├── models.py           # Model loading functions
│   │   ├── phonemes.py         # Phoneme extraction and comparison
│   │   └── speech.py           # Speech recognition
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   ├── gcs.py              # Google Cloud Storage functions
│   │   ├── phonemes.py         # Phoneme data and mappings
│   │   └── sentences.py        # Sentence database handling
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── audio_processing.py # Audio processing utilities
│       ├── text_processing.py  # Text processing utilities
│       ├── translations.py     # Translation utilities
│       └── session_state.py    # Session state management
├── tests/                      # Test files
├── .env.sample                 # Template for environment variables
├── .gitignore                  # Git ignore file
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── setup.py                    # Package configuration
└── README.md                   # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for inspiring the data organization
- [wav2vec2](https://huggingface.co/facebook/wav2vec2-base) and [Whisper](https://github.com/openai/whisper) for their speech processing models
- [Streamlit](https://streamlit.io/) for the easy-to-use web app framework
