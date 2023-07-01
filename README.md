# Microcontroller-With-Voice-AI-Jetson-
This repository is about my Final Project (Thesis) for Bachelor's Degree. This repo is about control light (bulb) that connect to Jetson with Voice. I using Wav2Vec2 and Audio Classification (Tensorflow) to Transcribe what speakers says and Verify the speakers.

# Step
1. Run Automatic Speech Recognition (Wav2Vec2_Transfer_Learning_Local.ipynb) to Create the Model
2. Run Audio Classifiation (AudioClassification.ipynb) to Create the Model
3. Convert Audio Classification Model into .onnx (For Performance purpose)
4. Run LiveASR.ipynb to make sure the both of models runs well
5. Run Monitoring.ipynb to check if Monitoring Web runs well
6. Put (asr_LM_skripsi_local_common_voice, asr_skripsi_local_common_voice, Classification.simplified.onnx, MonitorWeb, monitorWeb.py) into Jetson
7. Run monitorWeb.py on Jetson
8. Run ngrok on Jetson to Monitoring via other Devices 
