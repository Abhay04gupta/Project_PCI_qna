import os
import logging
from flask import Flask, request, send_file, make_response, render_template, url_for, jsonify
from werkzeug.utils import secure_filename
from gtts import gTTS
from transformers import pipeline
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import speech_recognition as sr
import numpy as np
import librosa

app = Flask(__name__)

# Define folder paths
UPLOAD_FOLDER_GEN = 'upload_gen'
UPLOAD_FOLDER_TEST = 'upload_test'
GENERATED_FOLDER_GEN = 'generated_gen'
GENERATED_FOLDER_TEST = 'generated_test'

app.config['UPLOAD_FOLDER_GEN'] = UPLOAD_FOLDER_GEN
app.config['UPLOAD_FOLDER_TEST'] = UPLOAD_FOLDER_TEST
app.config['GENERATED_FOLDER_GEN'] = GENERATED_FOLDER_GEN
app.config['GENERATED_FOLDER_TEST'] = GENERATED_FOLDER_TEST

use_test_folder = False

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER_GEN, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_TEST, exist_ok=True)
os.makedirs(GENERATED_FOLDER_GEN, exist_ok=True)
os.makedirs(GENERATED_FOLDER_TEST, exist_ok=True)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('qna.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    global use_test_folder
    global answer
    try:
        cleanup_folder(app.config['GENERATED_FOLDER_GEN'])
        
        if 'audio' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        

        storage_folder = app.config['UPLOAD_FOLDER_TEST'] if use_test_folder else app.config['UPLOAD_FOLDER_GEN']
        generated_folder = app.config['GENERATED_FOLDER_GEN']

        filename = secure_filename(file.filename)
        webm_path = os.path.join(storage_folder, filename)
        file.save(webm_path)
        
        mp3_path = rename_webm_to_mp3(storage_folder, filename)

        # Speech to text conversion
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small.en")
        question = pipe(mp3_path)["text"]
        
        if "help" in question.lower():
            use_test_folder = True
            answer = "Sure, choose any one topic from the following and talk about it for 30 seconds."
            
            # Text to speech conversion
            tts = gTTS(text=answer, lang='en')
            generated_filename = os.path.basename(mp3_path).replace('.mp3', '_generated.mp3')
            generated_path = os.path.join(generated_folder, generated_filename)
            tts.save(generated_path)

            # Create a URL for the generated audio
            audio_url = url_for('send_generated_file', filename=generated_filename)

            cleanup_folder(app.config['UPLOAD_FOLDER_GEN'])

            return jsonify({
                "question": "1) If you could choose one song to play every time you walked into a room, what would you choose and why?\n2) What do you think about when you're alone in a room?\n3) If you could have any superhero power, what would it be and why?\n4) What would you do if you were the only survivor of a plane crash?\n5) If you could eliminate one state, what would it be and why?",
                # "answer": answer,
                "audio_url": audio_url
                })
        
        elif use_test_folder:
            # This block is executed when the user speaks after "help"
            # Analyze audio only after help
            y, sr = librosa.load(mp3_path, sr=None)
            metrics = analyze_audio(y, sr)

            answer="Your Communication test result is on screen"
            
            # Text to speech conversion
            tts = gTTS(text=answer, lang='en')
            generated_filename = os.path.basename(mp3_path).replace('.mp3', '_generated.mp3')
            generated_path = os.path.join(app.config['GENERATED_FOLDER_GEN'], generated_filename)
            tts.save(generated_path)

            # Create a URL for the generated audio
            audio_url = url_for('send_generated_file', filename=generated_filename)

            use_test_folder=False
            cleanup_folder(app.config['UPLOAD_FOLDER_TEST'])
            cleanup_folder(app.config['UPLOAD_FOLDER_GEN'])

            return jsonify({
                # "question": question,
                # "answer": answer,
                "metrics": metrics,
                "audio_url": audio_url
            })

        else:
            # LLM processing for other cases
            model_cpt = "meta-llama/Llama-3.2-3B-Instruct"    #"tiiuae/falcon-7b-instruct"
            huggingface_api_token = "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"

            llm = HuggingFaceHub(
                repo_id=model_cpt,
                model_kwargs={"temperature": 0.7, "max_new_tokens": 500},
                huggingfacehub_api_token=huggingface_api_token
            )

            template = """Question: {question}\nAnswer:"""
            prompt = PromptTemplate(template=template, input_variables=['question'])
            chain = LLMChain(prompt=prompt, llm=llm)

            out = chain.run(question)
            answer = out.split("Answer:")[1].replace("User", "").split("\n")[0]

            # Text to speech conversion
            tts = gTTS(text=answer, lang='en')
            generated_filename = os.path.basename(mp3_path).replace('.mp3', '_generated.mp3')
            generated_path = os.path.join(generated_folder, generated_filename)
            tts.save(generated_path)

            # Create a URL for the generated audio
            audio_url = url_for('send_generated_file', filename=generated_filename)

            cleanup_folder(app.config['UPLOAD_FOLDER_GEN'])
            

            return jsonify({
                # "question": question,
                # "answer": answer,
                # "metrics": metrics,
                "audio_url": audio_url
            })
            

    except Exception as e:
        logging.error("Error processing audio: %s", str(e))
        return jsonify({"error": "An error occurred while processing the audio"}), 500


def rename_webm_to_mp3(folder, webm_filename):
    old_file_path = os.path.join(folder, webm_filename)
    new_filename = webm_filename.replace('.webm', '.mp3')
    new_file_path = os.path.join(folder, new_filename)
    
    if os.path.exists(old_file_path):
        os.rename(old_file_path, new_file_path)
    else:
        print(f'File "{old_file_path}" not found in "{folder}".')

    return new_file_path

def cleanup_folder(folder):
    """Function to remove all files in the specified folder."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Remove the directory if empty
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def analyze_audio(y, sr):
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Pitch estimation using piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    mean_pitch = np.mean(pitch) if len(pitch) > 0 else 0
    
    # Calculate RMS energy
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    mean_intensity = np.mean(rms)
    
    stressed_syllables_count = calculate_stressed_syllables(y)
    falling_intonation = detect_falling_intonation(y)
    detected_silence_sections = detect_silent_sections(y)
    number_of_words = count_words(y)
    speech_rate = calculate_speech_rate(number_of_words, total_duration)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    root_mean_square_energy = np.mean(rms)
    fluency_improvement_needed = len(detected_silence_sections) > 0

    # Tagging based on thresholds
    mean_pitch_tag = "Good" if mean_pitch > 1000  else "Bad"
    
    intensity_tag = "Loud" if mean_intensity > 0.05 else "Soft"
    
    stressed_syllables_tag = "Natural" if stressed_syllables_count < 70 else "Dramatic"
    
    falling_intonation_tag = "Need Confidence" if falling_intonation else "Good Confidence"
    
    silence_tag = "Good" if len(detected_silence_sections) == 0 else "Bad"
    
    word_count_tag = "Appropriate Speed" if number_of_words > 70 else "Speak a bit Quickly"
    
    speech_rate_tag = "Good" if speech_rate >= 100 else "Slow"
    
    tempo_tag = "Good" if tempo >= 90 and tempo <= 200 else "Bad"
    
    zero_crossing_rate_tag = "Good" if zero_crossing_rate < 0.07 else "Bad"
    
    rms_energy_tag = "Good" if root_mean_square_energy >= 0.04 else "Bad"
    
    fluency_tag = "Bad Fluency" if not fluency_improvement_needed else "Good Fluency"
    
    return {
        "total_duration":float(total_duration),
        "mean_pitch": {"value":float(mean_pitch),"tag":mean_pitch_tag},
        "mean_intensity": {"value":float(mean_intensity),"tag":intensity_tag},
        "stressed_syllables": {"value": int(stressed_syllables_count),"tag":stressed_syllables_tag},
        "falling_intonation": {"tag":falling_intonation_tag},
        "detected_silence_sections": {"value":len(detected_silence_sections),"tag":silence_tag},
        "number_of_words": {"value":int(number_of_words),"tag":word_count_tag},
        "speech_rate": {"value":float(speech_rate),"tag":speech_rate_tag},
        "fluency_tempo": {"value":float(tempo),"tag":tempo_tag},
        "zero_crossing_rate": {"value":float(zero_crossing_rate),"tag":zero_crossing_rate_tag},
        "root_mean_square_energy": {"value":float(root_mean_square_energy),"tag":rms_energy_tag},
        "fluency_improvement_needed": {"value":bool(fluency_improvement_needed),"tag":fluency_tag}
    }

def calculate_stressed_syllables(y):
    # Implement stress detection logic
    return len(librosa.onset.onset_detect(y=y, sr=22050, units='samples'))

def detect_falling_intonation(y):
    pitches, _ = librosa.piptrack(y=y, sr=22050)
    pitch_mean = np.mean(pitches, axis=1)
    return pitch_mean[-1] < pitch_mean[0]

def detect_silent_sections(y, threshold_db=-60, min_silence_duration=0.5):
    intervals = librosa.effects.split(y, top_db=abs(threshold_db))
    silent_sections = []
    for start, end in intervals:
        duration = librosa.samples_to_time(end - start)
        if duration >= min_silence_duration:
            silent_sections.append((start, end))
    return silent_sections

def count_words(y):
    # This is a rough estimation. For accurate word count, you'd need a full speech-to-text conversion.
    onsets = librosa.onset.onset_detect(y=y, sr=22050, units='time')
    return len(onsets)

def calculate_speech_rate(word_count, duration):
    return word_count / (duration / 60)  # words per minute

@app.route('/generated_gen/<path:filename>', methods=['GET'])
def send_generated_file(filename):
    return send_file(os.path.join(GENERATED_FOLDER_GEN, filename))

if __name__ == '__main__':
    app.run(debug=True)
