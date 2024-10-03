from flask import Flask, request, render_template, redirect, url_for
import whisper
from pydub import AudioSegment
from pyannote.audio import Pipeline
import re
import os

app = Flask(__name__)

# Ensure you have a folder for saving uploaded audio
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_file_path)

    # Convert the audio file to WAV format for compatibility with Whisper
    wav_path = os.path.join(UPLOAD_FOLDER, "conversation.wav")
    audio.export(wav_path, format="wav")

    # Transcribe the audio file to text
    result = model.transcribe(wav_path)

    # Return the transcribed text
    return result["text"]

# Function to perform speaker diarization using pyannote
def diarize_audio(audio_file_path):
    # Replace 'YOUR_AUTH_TOKEN' with your actual Hugging Face token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

    # Perform speaker diarization on the audio file
    diarization = pipeline(audio_file_path)

    # Store the speaker segments and return them
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {
            "speaker": speaker,
            "start_time": turn.start,
            "end_time": turn.end
        }
        speaker_segments.append(segment)
    return speaker_segments

# Function to transcribe and split sentences
def transcribe_and_format(audio_file_path):
    transcription = transcribe_audio(audio_file_path)

    # Split transcription by sentence-ending punctuation marks
    sentences = re.split(r'(?<=[.?!])\s+', transcription)

    # Clean up and format the sentences
    formatted_text = "\n".join([sentence.strip() for sentence in sentences if sentence.strip()])

    return formatted_text

# Route for homepage to handle both input and output
@app.route('/', methods=['GET', 'POST'])
def home():
    transcription = None
    speaker_segments = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded audio file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Transcribe and diarize the audio file
            transcription = transcribe_and_format(file_path)
            speaker_segments = diarize_audio(file_path)

    return render_template('index.html', transcription=transcription, speaker_segments=speaker_segments)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
