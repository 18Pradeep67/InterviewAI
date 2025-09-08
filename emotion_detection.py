import os
from transformers import pipeline
from pydub import AudioSegment

MODEL_PATH = "./wav2vec2_finetuned_model"
AUDIO_FILE_PATH = "D:/Fall 25-26/Project-I/Emotion detection/data/Prince.wav"  # Example audio file

def convert_opus_to_wav(opus_path):
    wav_path = os.path.splitext(opus_path)[0] + ".wav"
    audio = AudioSegment.from_file(opus_path)
    audio.export(wav_path, format="wav")
    return wav_path

def get_emotion_probabilities(audio_path):
    if audio_path.endswith(".opus"):
        print(f"Converting {audio_path} to WAV format...")
        audio_path = convert_opus_to_wav(audio_path)

    if not os.path.exists(MODEL_PATH):
        return {"error": f"Model folder not found at: {MODEL_PATH}"}
        
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found at: {audio_path}"}

    print(f"Loading emotion model from: {MODEL_PATH}")
    emotion_classifier = pipeline("audio-classification", model=MODEL_PATH)

    print(f"Analyzing audio file for emotions: {audio_path}")
    predictions = emotion_classifier(audio_path, top_k=6)
    
    probabilities = {p['label'].lower(): round(p['score'], 4) for p in predictions}
    
    return probabilities

def generate_transcript_whisper(audio_path, model_name="openai/whisper-small"):
    """
    Generates a transcript for the audio using Whisper.
    model_name can be: tiny, small, medium, large, etc.
    """
    from transformers import pipeline
    print(f"Loading Whisper model ({model_name}) for transcription...")
    whisper_pipeline = pipeline("automatic-speech-recognition", model=model_name)
    
    print(f"Generating transcript for: {audio_path}")
    transcript = whisper_pipeline(audio_path)
    
    return transcript.get("text", "")

if __name__ == "__main__":
    emotion_probs = get_emotion_probabilities(AUDIO_FILE_PATH)
    
    print("\n--- Emotion Analysis Complete ---")
    if "error" in emotion_probs:
        print(f"An error occurred: {emotion_probs['error']}")
    else:
        print(f"File: {os.path.basename(AUDIO_FILE_PATH)}")
        print("Emotion Probabilities:", emotion_probs)

    # Generate transcript using Whisper
    transcript = generate_transcript_whisper(AUDIO_FILE_PATH, model_name="openai/whisper-small")
    print("\n--- Transcript ---")
    print(transcript)
