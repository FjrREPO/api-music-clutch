from flask import Flask, request, jsonify
from flask_cors import CORS

from engine.core import AudioRecognizer


app = Flask(__name__)
CORS(app)

recognizer = AudioRecognizer(
    model_path="./models/final_audio_recognition_model.keras",
    metadata_path="./data/track_names.json",
)


@app.route("/api/search", methods=["POST"])
def search_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        predictions = recognizer.predict_track(file)

        return jsonify(predictions)

    except Exception as e:
        app.logger.error(f"Search failed: {str(e)}")
        return jsonify({"error": "Audio recognition failed", "details": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
