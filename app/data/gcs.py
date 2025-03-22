"""
Google Cloud Storage interaction for the Indonesian Pronunciation App.
"""

import streamlit as st
import os
import pandas as pd
import io
import tempfile
import json
from google.cloud import storage
from google.oauth2 import service_account

def initialize_gcs_client(bucket_name=None):
    """
    Initialize Google Cloud Storage client with proper authentication.

    Args:
        bucket_name: Name of the GCS bucket to use

    Returns:
        GCS client object or None if initialization fails
    """
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
)
        # Create GCS client with application default credentials
        client = storage.Client(credentials=credentials, project=credentials.project_id)

        if bucket_name:
            # Try to access the bucket to validate connection
            try:
                bucket = client.bucket(bucket_name)
                print(f"Accessing GCS bucket '{bucket_name}'") # Print to console
                # Test if bucket exists by listing a small number of blobs
                next(bucket.list_blobs(max_results=1), None)
                st.session_state.gcs_bucket_name = bucket_name
                return client
            except Exception as e:
                st.error(f"Error accessing GCS bucket '{bucket_name}': {e}")
                return None
        return client
    except Exception as e:
        st.error(f"Error initializing GCS client: {e}")
        st.markdown("""
        ### Authentication Error

        To use Google Cloud Storage, you need to:

        1. Make sure you have a service account with access to the bucket
        2. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your service account key file
        3. Or use another authentication method as described in the Google Cloud documentation

        For more information, see [Google Cloud Authentication](https://cloud.google.com/docs/authentication)
        """)
        return None

@st.cache_data
def download_blob_to_temp(bucket_name, blob_name):
    """
    Download a blob from GCS to a temporary file without displaying messages.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob (file) in the bucket

    Returns:
        Path to the temporary file, or None if download fails
    """
    try:
        if st.session_state.gcs_client is None:
            st.session_state.gcs_client = initialize_gcs_client(bucket_name)

        if st.session_state.gcs_client is None:
            return None

        # Get bucket and blob
        bucket = st.session_state.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if the blob exists
        if not blob.exists():
            return None

        # Create temporary file with appropriate extension
        file_extension = os.path.splitext(blob_name)[1]
        if not file_extension:  # If no extension, default to .wav
            file_extension = '.wav'

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.close()

        # Download to the temporary file
        blob.download_to_filename(temp_file.name)

        return temp_file.name
    except Exception:
        return None

@st.cache_data
def load_sentences_dataframe_from_gcs(bucket_name, tsv_path):
    """
    Load the DataFrame containing sentences and audio filenames from Google Cloud Storage.
    Handles file paths more robustly.

    Args:
        bucket_name: Name of the GCS bucket
        tsv_path: Path to the TSV file in the bucket

    Returns:
        DataFrame with sentences and audio information
    """
    gcp_credentials = dict(st.secrets["gcp_service_account"])

    try:
        # Check if GCS client is initialized
        if st.session_state.gcs_client is None:
            st.session_state.gcs_client = initialize_gcs_client(bucket_name)

        if st.session_state.gcs_client is None:
            return None

        # Get bucket and blob
        bucket = st.session_state.gcs_client.bucket(bucket_name)
        blob = bucket.blob(tsv_path)

        # Check if the blob exists
        if not blob.exists():
            st.error(f"TSV file not found in bucket: {tsv_path}")
            return None

        # Download TSV content to a string
        tsv_content = blob.download_as_string().decode('utf-8')

        # Determine if file is TSV or CSV based on content
        if '\t' in tsv_content.split('\n')[0]:
            # Read as TSV
            # df = pd.read_csv(io.StringIO(tsv_content), sep='\t', storage_options={"token": gcp_credentials})
            df = pd.read_csv("gs://natify/final_audio/filtered_results.tsv", sep='\t', storage_options={"token": gcp_credentials})
        else:
            # Read as CSV
            # df = pd.read_csv(io.StringIO(tsv_content), storage_options={"token": gcp_credentials})
            df = pd.read_csv("gs://natify/final_audio/filtered_results.tsv", storage_options={"token": gcp_credentials})

        # Make sure required columns exist
        required_columns = ['sentence', 'path']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Data file must contain these columns: {required_columns}")
            return None

        # Clean up the DataFrame
        df = df.dropna(subset=['sentence', 'path'])

        # Normalize paths - ensure they're valid for GCS
        # If paths don't start with '/', add it (common GCS issue)
        if 'path' in df.columns:
            # Extract the directory from the TSV path
            tsv_dir = os.path.dirname(tsv_path)

            # Function to fix relative paths
            def fix_path(path):
                # If the path is already absolute (starts with gs:// or /), return as is
                if path.startswith('gs://') or path.startswith('/'):
                    return path

                # If the path is relative, join it with the TSV directory
                if tsv_dir:
                    return os.path.join(tsv_dir, path)
                return path

            # Apply the fix to all paths
            df['path'] = df['path'].apply(fix_path)

        # Add difficulty column based on sentence length and complexity
        def determine_difficulty(sentence):
            words = len(sentence.split())
            if words <= 3:
                return "easy"
            elif words <= 8:
                return "medium"
            else:
                return "difficult"

        df['difficulty'] = df['sentence'].apply(determine_difficulty)

        # Add translation column if not present - silently add placeholders without showing a message
        if 'translation' not in df.columns:
            # Just add placeholder values - we'll translate on demand later
            df['translation'] = "Translation not available"

        return df

    except Exception as e:
        st.error(f"Error loading DataFrame from GCS: {e}")
        return None
