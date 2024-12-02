import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import json
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
import logging

class AudioRecognizer:
    def __init__(self, model_path=None, metadata_path=None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Default paths if not provided
        model_path = model_path or '../models/final_audio_recognition_model.keras'
        metadata_path = metadata_path or '../data/track_names.json'

        try:
            # Load pre-trained model with error handling
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            self.model = tf.keras.models.load_model(model_path)
            self.logger.info(f"Model loaded successfully from {model_path}")

            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.track_names = metadata["track_names"]
            self.label_mapping = metadata["label_mapping"]
            
            # Setup label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(self.track_names)

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise

    def convert_audio_to_standard_format(self, audio_file, target_dir='/tmp'):
        try:
            os.makedirs(target_dir, exist_ok=True)
            
            temp_input = os.path.join(target_dir, "input_audio.webm")
            temp_wav = os.path.join(target_dir, "output.wav")

            # Handle file-like objects and paths
            if hasattr(audio_file, 'read'):
                with open(temp_input, "wb") as f:
                    f.write(audio_file.read())
            else:
                temp_input = audio_file

            # Convert to WAV
            audio = AudioSegment.from_file(temp_input, format="webm")
            audio.export(temp_wav, format="wav")

            # Read WAV file
            audio_array, sample_rate = sf.read(temp_wav)

            # Clean up temporary files
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

            return audio_array, sample_rate

        except Exception as e:
            self.logger.error(f"Audio conversion error: {e}")
            raise

    def preprocess_audio(self, audio_array, sample_rate, target_duration=30):
        try:
            # Ensure mono audio
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            # Fixed duration processing
            target_length = sample_rate * target_duration

            if len(audio_array) > target_length:
                audio_array = audio_array[:target_length]
            else:
                audio_array = np.pad(
                    audio_array, 
                    (0, max(0, target_length - len(audio_array))), 
                    mode='constant'
                )

            # Z-score normalization
            audio_array = (audio_array - np.mean(audio_array)) / np.std(audio_array)
            return audio_array

        except Exception as e:
            self.logger.error(f"Audio preprocessing error: {e}")
            raise

    def extract_advanced_features(self, audio_array, sample_rate):
        try:
            features_list = [
                librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=96),
                librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13),
                librosa.feature.chroma_stft(y=audio_array, sr=sample_rate)
            ]

            # Convert to decibel scale and pad
            features = np.concatenate([
                librosa.power_to_db(feat) if feat.ndim > 1 else feat.reshape(1, -1)
                for feat in features_list
            ], axis=0)

            # Standardize to 300 time steps
            if features.shape[1] > 300:
                features = features[:, :300]
            else:
                padding = 300 - features.shape[1]
                features = np.pad(
                    features, 
                    ((0, 0), (0, padding)), 
                    mode='constant'
                )

            return features.T.reshape(1, 300, 96)

        except Exception as e:
            self.logger.error(f"Advanced feature extraction error: {e}")
            raise

    def predict_track(self, audio_file):
        try:
            # Convert audio to standard format
            audio_array, sample_rate = self.convert_audio_to_standard_format(audio_file)

            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_array, sample_rate)

            # Extract features
            features = self.extract_advanced_features(processed_audio, sample_rate)

            # Predict
            predictions = self.model.predict(features)[0]

            # Get top 3 predictions with track names and confidence
            top_3_indices = predictions.argsort()[-3:][::-1]
            top_3_tracks = [
                {
                    "name": self.track_names[idx],
                    "confidence": float(predictions[idx]),
                    "normalized_confidence": float(predictions[idx] / np.sum(predictions)) * 100
                }
                for idx in top_3_indices
            ]

            return top_3_tracks

        except Exception as e:
            self.logger.error(f"Track prediction error: {e}")
            raise