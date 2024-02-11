# %%
import os, subprocess, sys, json, shutil
from time import time
# from ai4bharat.transliteration import XlitEngine
import os
import whisper
import os
import whisperx
import gc, tqdm
from pydub import AudioSegment
import cv2
import numpy as np
import subprocess, json
import yt_dlp as youtube_dl
import requests
from time import time, sleep
IN_COLAB = 'google.colab' in sys.modules
# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

class lyric_extraction:
    def __init__(self, model_): # ignore the model - set based on platform
        self.device = 'cuda'
        self.batch_size = 16
        self.compute_type = 'float16' if IN_COLAB else 'int8'
        self.language_to_iso = {
            'Assamese': 'as',
            'Bengali': 'bn',
            'English': 'en',
            'Gujarati': 'gu',
            'Hindi': 'hi',
            'Kannada': 'kn',
            'Malayalam': 'ml',
            'Marathi': 'mr',
            'Punjabi': 'pa',
            'Sanskrit': 'si',
            'Tamil': 'ta',
            'Telugu': 'tg',
            'Urdu': 'ur',
        }
        for key, value in list(self.language_to_iso.items()):
            self.language_to_iso[key.lower()] = value
        self.model = whisperx.load_model('large-v2' if self.compute_type == 'float16' else 'medium', self.device, compute_type = self.compute_type)
    def run_vocal_extraction(self, track_path, language):
        print(f"Running Lyric Extraction [{language}]:", track_path)
        # res = whisper.transcribe(self.model, track_path)
        if len(language) > 2:
            language = self.language_to_iso[language]
        
        audio = whisperx.load_audio(track_path)
        print('Transcribing')
        audio = whisperx.load_audio(track_path)
        if language.strip().lower() != 'auto':
            res = self.model.transcribe(audio, language = language)
        else:
            res = self.model.transcribe(audio)
        print(res)
        lyrics = [
            {
                "start_time": segment['start'],
                "end_time": segment['end'],
                "lyric": segment['text']
            }
                for segment in res['segments']
        ]
        return lyrics

    
if __name__ == '__main__':
    lr = lyric_extraction('')
    res = lr.run_vocal_extraction('harvard.wav', language = 'en')
    print(json.dumps(res, indent = 4))