import os
import numpy as np
import librosa
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import List, Dict
import logging
from tensorflow import keras


class AudioMatcher:
    def __init__(
        self,
        model_path="./models/advanced_audio_recognition_model.keras",
        track_names_path="./data/track_names.json",
    ):
        # Load pre-trained model
        self.model = keras.models.load_model(model_path)

        # Load track names and label mapping
        with open(track_names_path, "r") as f:
            track_data = json.load(f)
            self.track_names = track_data["track_names"]
            self.label_mapping = track_data["label_mapping"]

        # Reverse the label mapping for easy decoding
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

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
        predictions = self.model.predict(fingerprint_input)[0]

        # Sort matches by probability in descending order
        sorted_indices = np.argsort(predictions)[::-1]

        # Prepare match results
        matches = []
        for idx in sorted_indices[:4]:  # Return top 4 matches
            if predictions[idx] > 0.7:  # Only include matches above 10% confidence
                matches.append(
                    {
                        "track": self.reverse_label_mapping[idx],
                        "confidence": float(predictions[idx]),
                    }
                )

        return matches


# Flask Application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize audio matcher
audio_matcher = AudioMatcher()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        audio_file.save(filepath)
        logger.info(f"File saved to {filepath}")

        try:
            # Match audio
            matches = audio_matcher.match_audio(filepath)
            logger.info(f"Audio matched: {matches}")

            # Clean up uploaded file
            # os.remove(filepath)

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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
