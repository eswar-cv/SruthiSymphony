import json
import os
import tqdm
import cv2
from PIL import Image
import numpy as np
import os
import random
import tqdm
import librosa 
from image_maker import *
import soundfile as sf
import IPython.display as ipd
import numpy as np
import math
import shutil
import IPython.display as ipd
import cv2
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import copy
from moviepy.editor import ImageSequenceClip
from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip
import datetime
from song_processor import song_processor
from video_generator import video_generator 
from evaluation_note import evaluation_note
# from evaluation_lyric import evaluation_lyric
class flow_manager:
    def __init__(self, lyrics_model):
        self.song_processing_agent = song_processor(lyrics_model = lyrics_model)
        self.video_maker = video_generator()

    def make_tutorial(self, fname, language, raga, user_notes = []):
        if not language:
            language = 'auto'
        fname_ = os.path.split(fname)[-1]
        song_details = self.song_processing_agent.process_song(fname_, language)
        # print("SONG DETAILS", song_details)
        # print(song_details)
        audio_length = AudioSegment.from_file(fname).duration_seconds
        song_details['user_notes'] = user_notes
        song_details['raga'] = raga
        video_location = self.video_maker.make_demo_video(fname, song_details, audio_length, 1)
        return video_location, song_details
    
    def make_report(self, fname, base_song_data):
        fname_ = os.path.split(fname)[-1]
        song_details = self.song_processing_agent.process_song(fname_)
        # print("SONG DETAILS", song_details)
        # print(song_details)
        audio_length = AudioSegment.from_file(fname).duration_seconds
        song_details['user_notes'] = base_song_data
        video_location = self.video_maker.make_demo_video(fname, song_details, audio_length, 1)
        return video_location, song_details
    
if __name__ == '__main__':
    flow_mgr = flow_manager()
    location = flow_mgr.make_tutorial("temp_files/english_sample_mini.wav")
    print(location)
