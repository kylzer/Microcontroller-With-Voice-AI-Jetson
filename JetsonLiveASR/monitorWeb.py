#!/usr/bin/env python

# Import Library
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, Wav2Vec2Processor
from flask import Flask, render_template, request, session, jsonify
import speech_recognition as sr
import torch
import gc
import librosa
import numpy as np
import onnxruntime as rt
import time
from flask_socketio import SocketIO, emit
import os
import Jetson.GPIO as GPIO


# Necessary Variable
relayCH4 = 11
relayCH3 = 13
relayCH2 = 15

SAMPLE_RATE = 16_000
FFT_SIZE = 2048
HOP_SIZE = 512
N_MELS = 26
MFCC_BINS = 13
MAX_LEN_PAD = 32
CHANNELS = 3
async_mode = None
labels = ['Adika Rajendra Haris', 'Banyu Ontoseno', 'Gesang Budiono']

# Set Controller

GPIO.setmode(GPIO.BOARD)
GPIO.setup(relayCH4, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(relayCH3, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(relayCH2, GPIO.OUT, initial=GPIO.HIGH)
# Necessary Function

# class StandardScaler(object):
#     def __init__(self):
#         pass

#     def fit(self, X):
#         self.mean_ = np.mean(X, axis=0)
#         self.scale_ = np.std(X - self.mean_, axis=0)
#         return self

#     def transform(self, X):
#         return (X - self.mean_) / self.scale_

#     def fit_transform(self, X):
#         return self.fit(X).transform(X)

def resample_audio(path):
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
    if len(audio) < SAMPLE_RATE: 
        audio = np.pad(audio, (0,16000-len(audio)), "constant")
    else:
        audio = audio[:SAMPLE_RATE]
    return audio

def stft(audio):
    audio_stft = librosa.stft(y = audio, n_fft = FFT_SIZE, hop_length = HOP_SIZE, center=True, pad_mode='constant')
    amp = np.abs(audio_stft)**2

    return amp

def mel_frequency(spectrogram):
    mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=N_MELS)
    
    mel_spec = mel.dot(spectrogram)
    db_mel_spect = librosa.power_to_db(mel_spec, ref=np.max)
    
    return db_mel_spect

def mfcc(mel_freq):
    mfcc = librosa.feature.mfcc(S=mel_freq, sr=SAMPLE_RATE, n_mfcc=MFCC_BINS, n_fft=FFT_SIZE, hop_length=HOP_SIZE, n_mels=N_MELS)
    
    if (MAX_LEN_PAD > mfcc.shape[1]):
        pad = MAX_LEN_PAD - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :MAX_LEN_PAD]
    return mfcc

def delta_mfcc(mfcc):
    delta_mfcc = librosa.feature.delta(mfcc)
    return delta_mfcc

def delta_delta_mfcc(mfcc):
    delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)
    return delta_delta_mfcc

def speaker_input_preprocessing(audio):
    audio_speaker = resample_audio("microphone-results.wav")
    audio_speaker = stft(audio_speaker)
    audio_speaker = mel_frequency(audio_speaker)

    audio_speaker_mfcc = mfcc(audio_speaker)
    audio_speaker_delta = delta_mfcc(audio_speaker_mfcc)
    audio_speaker_delta_delta = delta_delta_mfcc(audio_speaker_mfcc)

    audio_speaker_input = np.zeros((1, MFCC_BINS, MAX_LEN_PAD, CHANNELS), dtype=np.float32)
    # audio_speaker_input[:, :, :, 0] = std_scaler_inference.fit_transform(audio_speaker_mfcc)
    # audio_speaker_input[:, :, :, 1] = std_scaler_inference.fit_transform(audio_speaker_delta)
    # audio_speaker_input[:, :, :, 2] = std_scaler_inference.fit_transform(audio_speaker_delta_delta)
    audio_speaker_input[:, :, :, 0] = audio_speaker_mfcc
    audio_speaker_input[:, :, :, 1] = audio_speaker_delta
    audio_speaker_input[:, :, :, 2] = audio_speaker_delta_delta
    return audio_speaker_input


# Model Apply

model = Wav2Vec2ForCTC.from_pretrained("asr_skripsi_local_common_voice/checkpoint-2400").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("asr_skripsi_local_common_voice/")
processorLM = Wav2Vec2ProcessorWithLM.from_pretrained("asr_LM_skripsi_local_common_voice", eos_token=None, bos_token=None)
model_onnx = rt.InferenceSession('Classification.simplified.onnx', providers=["CUDAExecutionProvider"])


# Initialize Speech Recognizer
listener = sr.Recognizer()
listener.dynamic_energy_threshold = True
# listener.energy_threshold = 4500


# Flask (HTML)
app = Flask(__name__, template_folder='MonitoringWeb', static_folder='MonitoringWeb/static')
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*", headers = {"Content-Type": "application/json; charset=utf-8"})
audio = ""
audio_speaker_input = ""
_ = ""

@app.route('/',methods=["GET", "POST"])
def index():
    return render_template('index.html', progress = "Wait For Record")

@app.route('/listen', methods=['POST'])
def record():
    listener = sr.Recognizer()
    listener.dynamic_energy_threshold = True
    data = request.get_json()
    print(data)
    if 'record' in data:
        try:
            
            with sr.Microphone(device_index=11) as mic:
                listener.adjust_for_ambient_noise(mic, duration=5)
                print("Listening...")
                audio = listener.listen(mic, phrase_time_limit=5)
                with open("microphone-results.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                return jsonify({'path': os.path.abspath("microphone-results.wav")})
            
        except sr.UnknownValueError:
            listener = sr.Recognizer()


@app.route('/preprocessing', methods=['POST'])
def preprocessing():
    data = request.get_json()['audioPath']
    print(data)

    global audio
    global audio_speaker_input
    global _
    
    audio, _ = librosa.load(data, sr=SAMPLE_RATE)
    audio_len = librosa.get_duration(y=audio, sr=SAMPLE_RATE)

    audio_speaker_input = speaker_input_preprocessing(audio)

    if audio_len > 5:
        print("Audio Duration More Than 5 Secs")
        return jsonify({'status': "Audio Duration More Than 5 Secs"})
    return jsonify({'status': "PreProcessing Done!"})

@app.route('/speaker-predict', methods=['POST'])
def speaker_predicting():
    print("Speaker Predicting...\n")
    inputDetails = model_onnx.get_inputs()
    start_time = time.time()
    pred_speaker = model_onnx.run(None, {inputDetails[0].name: audio_speaker_input})[0] 
    print("--- %s seconds ---" % (time.time() - start_time))

    if max(pred_speaker[0]) < 0.90:
        print("Speaker Unidentified")
        return jsonify({'speaker': "Speaker Unidentified"})
    top_index = np.argmax(pred_speaker[0])
    print(labels[top_index])
    return jsonify({'speaker': labels[top_index]})
    

@app.route('/word-predict', methods=['POST'])
def word_predicting():
    global audio
    
    print("\n\nWords Predicting...\n")
    start_time = time.time()
    input_dict = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(input_dict.input_values.to("cuda")).logits
    transcriptionLM = processorLM.batch_decode(logits.cpu().detach().numpy()).text[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"\n\nAudio Transcribe : {transcriptionLM}")
    
    if("nyalakan lampu satu nol satu" in transcriptionLM):
        print("Lampu satu nol satu nyala")
        GPIO.output(relayCH4, GPIO.LOW)
    elif("matikan lampu satu nol satu" in transcriptionLM):
        print("Lampu satu nol satu mati")
        GPIO.output(relayCH4, GPIO.HIGH)

    elif("nyalakan lampu satu nol dua" in transcriptionLM):                                                                     
        print("Lampu satu nol dua nyala")                                                                                      
        GPIO.output(relayCH3, GPIO.LOW) 
    elif("matikan lampu satu nol dua" in transcriptionLM):                                                                     
        print("Lampu satu nol dua mati")                                                                                      
        GPIO.output(relayCH3, GPIO.HIGH) 

    elif("nyalakan lampu dua nol satu" in transcriptionLM):                                                                     
        print("Lampu dua nol satu nyala")                                                                                      
        GPIO.output(relayCH2, GPIO.LOW) 
    elif("matikan lampu dua nol satu" in transcriptionLM):                                                                     
        print("Lampu dua nol satu mati")                                                                                      
        GPIO.output(relayCH2, GPIO.HIGH) 
        
    del audio, input_dict, logits
    gc.collect()
    torch.cuda.empty_cache()
    return jsonify({"transcribe" : transcriptionLM})

if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
