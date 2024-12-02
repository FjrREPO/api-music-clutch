import os
import numpy as np
import librosa
import json
import sys
import logging

# Explicitly handle TensorFlow and Keras import
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow or Keras import failed. Attempting alternative import.")
    try:
        import keras
    except ImportError:
        print("Could not import Keras. Please check your installation.")
        sys.exit(1)

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AudioMatcher:
    def __init__(
        self,
        model_path=os.getenv('MODEL_PATH', "./models/advanced_audio_recognition_model.keras"),
        track_names_path=os.getenv('TRACK_NAMES_PATH', "./data/track_names.json"),
    ):
        # Configure TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        
        # Load pre-trained model with enhanced error handling
        try:
            # Try multiple import methods
            try:
                self.model = tf.keras.models.load_model(model_path)
            except Exception as tf_load_error:
                try:
                    self.model = keras.models.load_model(model_path)
                except Exception as keras_load_error:
                    logging.error(f"Model loading failed. TensorFlow Error: {tf_load_error}")
                    logging.error(f"Keras Error: {keras_load_error}")
                    raise RuntimeError("Could not load the model using TensorFlow or Keras")

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

        # Load track names and label mapping
        try:
            with open(track_names_path, "r") as f:
                track_data = json.load(f)
                self.track_names = track_data["track_names"]
                self.label_mapping = track_data["label_mapping"]

            # Reverse the label mapping for easy decoding
            self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        except Exception as e:
            logging.error(f"Failed to load track names: {e}")
            raise

    def generate_fingerprint(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Generate a robust fingerprint from audio data"""
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        audio_array = librosa.util.normalize(audio_array)

        features = [
            librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=64),
            librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=20),
            librosa.feature.chroma_stft(y=audio_array, sr=sample_rate),
            librosa.feature.spectral_contrast(y=audio_array, sr=sample_rate),
        ]

        processed_features = [
            librosa.power_to_db(feat) if feat.ndim > 1 else feat.reshape(1, -1)
            for feat in features
        ]

        combined_features = np.concatenate(processed_features, axis=0)
        fingerprint = np.mean(combined_features, axis=1)

        # Ensure the fingerprint has exactly 104 elements
        if fingerprint.shape[0] != 104:
            # If too short, pad with zeros
            if fingerprint.shape[0] < 104:
                padded_fingerprint = np.zeros(104)
                padded_fingerprint[: fingerprint.shape[0]] = fingerprint
                fingerprint = padded_fingerprint
            # If too long, truncate
            else:
                fingerprint = fingerprint[:104]

        return fingerprint

    def match_audio(self, audio_path: str) -> List[Dict[str, float]]:
        """Match input audio against trained model"""
        # Load audio file
        audio_array, sample_rate = librosa.load(audio_path, sr=None)

        # Generate fingerprint
        fingerprint = self.generate_fingerprint(audio_array, sample_rate)

        # Reshape to match model's expected input
        fingerprint_input = fingerprint.reshape(1, -1)

        # Predict probabilities
        # Use the appropriate prediction method based on available library
        try:
            predictions = self.model.predict(fingerprint_input)[0]
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise

        # Sort matches by probability in descending order
        sorted_indices = np.argsort(predictions)[::-1]

        # Prepare match results
        matches = []
        for idx in sorted_indices[:4]:  # Return top 4 matches
            if predictions[idx] > 0.7:  # Only include matches above 70% confidence
                matches.append(
                    {
                        "track": self.reverse_label_mapping[idx],
                        "confidence": float(predictions[idx]),
                    }
                )

        return matches

# Flask Application Configuration
def create_app():
    app = Flask(__name__)
    
    # Configure CORS with specific origins if needed
    CORS(app, resources={
        r"/api/*": {
            "origins": os.getenv('ALLOWED_ORIGINS', '*').split(',')
        }
    })

    # Configure upload directory with absolute path
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/audio_uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    # Set maximum file size (10 MB)
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.getenv('LOG_FILE', 'audio_matcher.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Initialize audio matcher
    try:
        audio_matcher = AudioMatcher()
    except Exception as e:
        logger.error(f"Failed to initialize AudioMatcher: {e}")
        raise

    @app.route("/api/search", methods=["POST"])
    def search_audio():
        if "audio" not in request.files:
            logger.error("No audio file uploaded")
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files["audio"]

        if audio_file.filename == "":
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if audio_file:
            # Secure filename and save
            filename = secure_filename(audio_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            try:
                audio_file.save(filepath)
                logger.info(f"File saved to {filepath}")

                # Match audio
                matches = audio_matcher.match_audio(filepath)
                logger.info(f"Audio matched: {matches}")

                # Clean up uploaded file
                os.remove(filepath)

                return jsonify(matches), 200

            except Exception as e:
                # Clean up file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                logger.error(f"Audio processing failed: {e}")

                return jsonify({"error": "Audio processing failed", "details": str(e)}), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({"error": "File too large"}), 413

    return app

# For Gunicorn or other WSGI servers
app = create_app()

# For development server
if __name__ == "__main__":
    app = create_app()
    app.run(
        host=os.getenv('HOST', '0.0.0.0'), 
        port=int(os.getenv('PORT', 5000)), 
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )