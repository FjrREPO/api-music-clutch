import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import json
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment


class AudioRecognizer:
    def __init__(self, model_path, metadata_path):
        self.model = tf.keras.models.load_model(model_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.track_names = metadata["track_names"]
        self.label_mapping = metadata["label_mapping"]
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(self.track_names)

    def convert_webm_to_wav(self, webm_file):
        try:
            os.makedirs("/tmp", exist_ok=True)
            
            temp_webm = "/tmp/input.webm"
            temp_wav = "/tmp/output.wav"

            with open(temp_webm, "wb") as f:
                f.write(webm_file.read())

            audio = AudioSegment.from_file(temp_webm, format="webm")
            audio.export(temp_wav, format="wav")

            audio_array, sample_rate = sf.read(temp_wav)

            os.remove(temp_webm)
            os.remove(temp_wav)

            return audio_array, sample_rate

        except Exception as e:
            print(f"Audio conversion error: {e}")
            raise

    def convert_audio_to_array(self, audio_array, sample_rate):
        try:
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            target_length = sample_rate * 30
            if len(audio_array) > target_length:
                audio_array = audio_array[:target_length]
            else:
                audio_array = np.pad(
                    audio_array, (0, target_length - len(audio_array)), "constant"
                )

            audio_array = (audio_array - np.mean(audio_array)) / np.std(audio_array)
            return audio_array

        except Exception as e:
            print(f"Audio conversion error: {e}")
            raise

    def extract_features(self, audio_array, sample_rate, max_pad_len=200):
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio_array, sr=sample_rate, n_mels=128, fmax=sample_rate // 2
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            mfcc = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)

            features = np.concatenate([mel_spec_db, mfcc], axis=0)

            if features.shape[1] > max_pad_len:
                features = features[:, :max_pad_len]
            else:
                padding = max_pad_len - features.shape[1]
                features = np.pad(features, ((0, 0), (0, padding)), mode="constant")

            return features.T

        except Exception as e:
            print(f"Feature extraction error: {e}")
            raise

    def predict_track(self, audio_file):
        try:
            audio_array, sample_rate = self.convert_webm_to_wav(audio_file)

            audio_array = self.convert_audio_to_array(audio_array, sample_rate)

            features = self.extract_features(audio_array, sample_rate)

            features = np.expand_dims(features, axis=0)

            predictions = self.model.predict(features)[0]

            top_3_indices = predictions.argsort()[-3:][::-1]
            top_3_tracks = [
                {"name": self.track_names[idx], "confidence": float(predictions[idx])}
                for idx in top_3_indices
            ]

            return top_3_tracks

        except Exception as e:
            print(f"Prediction error: {e}")
            raise