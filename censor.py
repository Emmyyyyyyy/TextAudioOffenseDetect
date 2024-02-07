import re
from pydub import AudioSegment
from pydub.playback import play
from tkinter import ttk
import wave

import numpy as np

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from tkinter import filedialog, simpledialog

#----------# audio #----------# 

mp3_beep = AudioSegment.from_mp3('./audio/bleep.mp3')
beep = mp3_beep.set_frame_rate(44100).set_channels(1).set_sample_width(2)

export_format = None

def beep_audio(duration):
    return beep.speedup(playback_speed=len(beep) / (duration * 1000))
                
def audio_cut(audio, ranges_to_cut):
    remaining_audio = AudioSegment.silent(duration=0)
    
    for i, rng in enumerate(ranges_to_cut):
        start_time = int(rng["start"] * 1000)
        end_time = int(rng["end"] * 1000)
        if i == 0:
            remaining_audio += audio[:start_time]
            
        remaining_audio += beep_audio(rng["end"] - rng["start"])
        
        if i != len(ranges_to_cut) - 1:
            next_start_time = int(ranges_to_cut[i + 1]["start"] * 1000)
            remaining_audio += audio[end_time:next_start_time]
        else:
            remaining_audio += audio[end_time:]
    
    return remaining_audio

def choose_export_audio_type():
    global export_audio_format
    export_audio_format = simpledialog.askstring("Export Audio Format", "Enter export audio format (e.g., wav):")
    return export_audio_format

def export_censored_audio(censored_audio):
    global export_audio_format
    if not censored_audio:
        return

    if not export_audio_format:
        export_audio_format = "wav"

    file_path = filedialog.asksaveasfilename(defaultextension=f".{export_audio_format}", filetypes=[(f"{export_audio_format.upper()} files", f"*.{export_audio_format}")])
    censored_audio.export(file_path, format=export_audio_format)


#----------# text #----------# 

# def censor_words_in_sentence(sentence, words_to_censor):
#     words = sentence.split()
#     censored_sentence = ' '.join(censor_word(word, words_to_censor) for word in words)
#     # censored_sentence = censor_word(sentence, words_to_censor)

#     # print(censored_sentence)
#     return censored_sentence

def censor_words_in_sentence(sentence, phrases_to_censor):
    for phrase_info in phrases_to_censor:
        word = phrase_info['word']
        regex_pattern = re.compile(r'\b(?:' + re.escape(word) + r'|'
                                   + '|'.join(re.escape(word) + form for form in ["", "s", "ed", "ing"])
                                   + r')\b', re.IGNORECASE)
        sentence = regex_pattern.sub('[*censored*]', sentence)

    return sentence

def choose_format():
    global export_format
    export_format = simpledialog.askstring("Export Format", "Enter export format (e.g., txt):")
    return export_format

def export_censored_text(censored_text):
    global export_format
    if not censored_text:
        return

    if not export_format:
        export_format = "txt"

    file_path = filedialog.asksaveasfilename(defaultextension=f".{export_format}", filetypes=[(f"{export_format.upper()} files", f"*.{export_format}")])
    with open(file_path, "w") as file:
        file.write(censored_text)

def export_words_to_censor(words_to_censor_data):
    if not words_to_censor_data:
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    with open(file_path, "w") as file:
        for entry in words_to_censor_data:
            file.write(f"{entry['word']}, {entry['start']}, {entry['end']}\n")

def censor_word(word, words_to_censor):
    for entry in words_to_censor:
        if entry["word"] in word or word in entry["word"]:
            censored_word = entry.get("censored", censor_default(entry["word"]))
            return word.replace(entry["word"], censored_word)
    return word

def censor_default(word):
    if len(word) < 3:
        return word[0]
    elif len(word) == 2:
        return word[0] + '*'
    else:
        censored_word = word[0] + '*' * (len(word) - 2) + word[-1]
        return censored_word

# sentence = "fuck that dick"
# censored_result = censor_words_in_sentence(sentence, [{"word": 'fuck', "start": 0.0, "end": 0.0}, {"word": 'dick', "start": 0.0, "end": 0.0}])
# print(censored_result)