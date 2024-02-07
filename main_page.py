import os
import pickle
import tokenize
import wave
import time
import threading
import tkinter as tk
from tkinter import filedialog
from tkmacosx import Button
import pyaudio
from pydub import AudioSegment
from speech_to_text import speech_to_text
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from pydub.playback import play
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

from word_model import censor_offensive_words
from censor import audio_cut
from censor import censor_words_in_sentence
from censor import export_censored_text
from censor import export_words_to_censor
from censor import choose_format
from censor import choose_export_audio_type
from censor import export_censored_audio

class MainPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Recorder")
        self.root.resizable(False, False)
        self.root.configure(bg='black')

        self.heading_label = tk.Label(self.root, text="Voice Recorder", font=('Inter', 60, 'bold'), bg='black')
        self.heading_label.grid(row=0, column=0, columnspan=2, pady=20, padx=20)

        self.choose_label = tk.Label(self.root, text="Choose Option", font=('Inter', 30, 'bold'), bg='black', fg='white')
        self.choose_label.grid(row=1, column=0, columnspan=2, pady=20)

        self.audio_button = Button(self.root, text='Audio', bg='orange', fg='white', borderless=1, font=('Inter', 15, 'bold'),
                                   command=lambda: self.open_audio_page(self.root))
        self.audio_button.grid(row=2, column=0, pady=10, padx=20)

        self.text_button = Button(self.root, text='Text', bg='blue', fg='white', borderless=1,
                                         font=('Inter', 15, 'bold'), command=lambda: self.open_text_page(self.root))
        self.text_button.grid(row=2, column=1, pady=10, padx=20)

    def open_audio_page(self, root):
        root.destroy()
        audio_page = tk.Tk()
        audio_page.title("Audio Page")
        audio_page.resizable(False, False)
        audio_page.configure(bg='black')
        self.text_entry = ""

        self.button = Button(audio_page, text='Click to start recording', bg='green', fg='white', borderless=1, font=('Inter', 30, 'bold'), 
                    command=self.click_handler)
        self.button.grid(row=0, column=0, pady=10, padx=20)

        self.choose_file_button = Button(audio_page, text='Choose File', bg='blue', fg='white', borderless=1,
                                        font=('Inter', 15, 'bold'), command=self.choose_file)
        self.choose_file_button.grid(row=1, column=0, pady=10)

        self.label = tk.Label(audio_page, text="00:00:00", bg='black', font='Inter')
        self.label.grid(row=0, column=1, pady=10)

        back_button = Button(audio_page, text='Back to Choose Option', bg='gray', fg='white', borderless=1,
                            font=('Inter', 15, 'bold'), command=lambda: self.back_to_main_page(audio_page))
        back_button.grid(row=2, column=0, pady=10)

        self.recording = False
        audio_page.mainloop()

    def open_text_page(self, root):
        root.destroy()
        text_page = tk.Tk()
        text_page.title("Text Page")
        text_page.resizable(False, False)
        text_page.configure(bg='black')

        self.text_entry = tk.Entry(text_page, font=('Inter', 15), bg='white', fg='black', width=35)
        self.text_entry.grid(row=0, column=0, pady=10, padx=20)

        self.send_button = Button(text_page, text='Send', bg='orange', fg='white', borderless=1, font=('Inter', 15, 'bold'),
                                command=self.send_text)
        self.send_button.grid(row=0, column=1, pady=10, padx=20)

        self.choose_file_button = Button(text_page, text='Choose File', bg='blue', fg='white', borderless=1,
                                        font=('Inter', 15, 'bold'), command=self.choose_file)
        self.choose_file_button.grid(row=1, column=0, pady=10)

        back_button = Button(text_page, text='Back to Choose Option', bg='gray', fg='white', borderless=1,
                            font=('Inter', 15, 'bold'), command=lambda: self.back_to_main_page(text_page))
        back_button.grid(row=2, column=0, pady=10)

        text_page.mainloop()

    def back_to_main_page(self, page):
        page.destroy()
        root = tk.Tk()
        main_page = MainPage(root)
        root.mainloop()

    def send_text(self):
        self.input_text = self.text_entry.get()
        # print("Input Text:", self.input_text)
        self.predict_text(self.input_text, 'text', [])
        self.text_entry.delete(0, tk.END)

    def choose_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.ogg *.flac *.aac *.m4a *.wma"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path = file_path
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.txt':
                with open(self.file_path, 'r') as text_file:
                    self.input_text = text_file.read()
                    print("Text content:", self.input_text)
                    self.predict_text(self.input_text, 'text', [])
            else:
                speech_to_text_result = speech_to_text(self.file_path)
                transcript = speech_to_text_result[len(speech_to_text_result)-1]['transcript']
                print("Speech to text result:", speech_to_text_result)
                sound = AudioSegment.from_file(f"{file_path}")
                self.sound = sound.set_frame_rate(44100).set_channels(1).set_sample_width(2)
                self.predict_text(transcript, 'audio', speech_to_text_result[:-1])

    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button.config(text="Click to start recording", bg='green')
        else:
            self.recording = True
            self.button.config(text="Click to stop recording", bg='red')
            threading.Thread(target=self.record).start()

    def record(self):
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []
        start = time.time()

        while self.recording:
            data = stream.read(1024)
            frames.append(data)

            passed = time.time() - start
            sec = passed % 60
            mins = passed // 60
            hours = mins // 60

            self.label.config(text=f'{int(hours):02d}:{int(mins):02d}:{int(sec):02d}')

        stream.stop_stream()
        stream.close()
        self.audio.terminate()

        exists = True
        i = 0
        while exists:
            if os.path.exists(f"recording{i}.wav"):
                i += 1
            else:
                exists = False
        
        sound_file = wave.open(f"recording{i}.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()

        result = speech_to_text(f"recording{i}.wav")
        transcript = result[len(result)-1]['transcript']
        print("Speech to text result:", result)
        self.sound = AudioSegment.from_wav(f"recording{i}.wav")
        self.predict_text(transcript, 'audio', result[:-1])

    def predict_text(self, input_text, type, timestamp):
        sentence_model = load_model('sentence_model_bi_lstm.h5')
        input_text_without_stopwords = self.custom_remove_stop_words(input_text)

        word_index_path = "word_index.pkl"
        with open(word_index_path, 'rb') as word_index_file:
            word_index = pickle.load(word_index_file)

        tokenizer = Tokenizer(num_words=4000, char_level=False, oov_token="<OOV>")
        tokenizer.word_index = word_index

        input_sequence = tokenizer.texts_to_sequences([input_text_without_stopwords])
        input_padded = pad_sequences(input_sequence, maxlen=100, padding='post', truncating='post')

        self.prediction = sentence_model.predict(input_padded)
        print("Sentence Prediction:", self.prediction)

        if self.prediction >= 0.8:
            self.words_to_censor = censor_offensive_words(input_text, timestamp)
            self.censored_sentence = self.censor_text(input_text, self.words_to_censor)

            if type == 'text':
                self.show_text_comparison(input_text, self.censored_sentence)
            elif type == 'audio':
                self.words_to_censor = [entry for entry in self.words_to_censor if entry['start'] != 0.0 or entry['end'] != 0.0]
                self.censor_audio(self.sound, self.words_to_censor, input_text, self.censored_sentence)
        else:
            if type == 'text':
                self.show_text_comparison(input_text, input_text)
            elif type == 'audio':
                self.plot_matplotlib_graph(self.sound, self.sound, input_text, input_text)
        
    def censor_text(self, sentence, words_to_censor):
        censored_sentence = censor_words_in_sentence(sentence, words_to_censor)
        # print(censored_sentence)
        return censored_sentence

    def censor_audio(self, audio, words_to_censor, input_text, censored_sentence):
        censored_audio = audio_cut(audio, words_to_censor)
        self.plot_matplotlib_graph(audio, censored_audio, input_text, censored_sentence)
        # return censored_audio

    def show_text_comparison(self, original, censored):
        
        text_root = tk.Toplevel()
        text_root.title("Compare sentence")

        classification_text = 'Offensive' if self.prediction > 0.8 else 'Not offensive'
        classification_color = 'red' if self.prediction > 0.8 else 'green'

        self.classification = tk.Label(text_root, text=classification_text, fg=classification_color, font=('Inter', 30, 'bold'))
        self.classification.pack(pady=10)

        label_original = tk.Label(text_root, text="Original Sentence:")
        label_original.pack()

        text_original = tk.Text(text_root, height=20, width=80)
        text_original.insert(tk.END, original)
        text_original.pack()

        label_censored = tk.Label(text_root, text="Censored Sentence:")
        label_censored.pack()

        text_censored = tk.Text(text_root, height=20, width=80)
        text_censored.insert(tk.END, censored)
        text_censored.pack()

        self.export_format_label = tk.Label(text_root, text="Chosen Format: .txt")
        self.export_format_label.pack()

        button_choose_format = Button(text_root, text="Choose Format", command=self.display_chosen_text_format)
        button_choose_format.pack()

        button_export_text = Button(text_root, text="Export Censored Text", command=lambda: export_censored_text(censored))
        button_export_text.pack()

        button_export_words_to_censor = Button(text_root, text="Export Words to Censor", command=lambda: export_words_to_censor(self.words_to_censor))
        button_export_words_to_censor.pack()

        button_close = Button(text_root, text="Close", command=lambda: [self.close_text_window(), text_root.destroy()])
        button_close.pack()

        text_root.mainloop()

    def plot_matplotlib_graph(self, audio, remaining_audio, input, censored_sentence):
        raw_audio1 = np.array(audio.get_array_of_samples(), dtype=np.int16)
        raw_audio2 = np.array(remaining_audio.get_array_of_samples(), dtype=np.int16)

        # play(audio)
        # play(remaining_audio)

        min_length = min(len(raw_audio1), len(raw_audio2))
        time = np.arange(0, min_length) / audio.frame_rate

        comparison_window = tk.Toplevel()
        comparison_window.title("Compare Sounds")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(time, raw_audio1[:min_length], label='Sound 1', color='blue')
        ax.plot(time, raw_audio2[:min_length], label='Sound 2', color='red', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=comparison_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)

        play_original_button = Button(comparison_window, text="Play original audio", command=lambda: play(audio))
        play_original_button.grid(row=1, column=0, pady=10, padx=10)

        play_remaining_button = Button(comparison_window, text="Play censored audio", command=lambda: play(remaining_audio))
        play_remaining_button.grid(row=1, column=1, pady=10, padx=10)

        self.export_audio_format_label = tk.Label(comparison_window, text="Chosen Format: .wav")
        self.export_audio_format_label.grid(row=2, column=0)

        button_choose_format = Button(comparison_window, text="Choose Format", command=self.display_chosen_audio_format)
        button_choose_format.grid(row=3, column=0)

        button_export_text = Button(comparison_window, text="Export Censored Audio", command=lambda: export_censored_audio(remaining_audio))
        button_export_text.grid(row=4, column=0)

        button_close = Button(comparison_window, text="Close", command=lambda: [comparison_window.destroy()])
        button_close.grid(row=5, column=0)

        comparison_window.after(100, lambda: self.show_text_comparison(input, censored_sentence))

        comparison_window.mainloop()

    def close_text_window(self):
        if self.text_entry:
            self.text_entry.delete(0, tk.END)

    def display_chosen_text_format(self):
        chosen_format = choose_format()
        if self.export_format_label is not None:
            self.export_format_label.config(text=f"Chosen Format: {chosen_format}")
    
    def display_chosen_audio_format(self):
        chosen_format = choose_export_audio_type()
        if self.export_audio_format_label is not None:
            self.export_audio_format_label.config(text=f"Chosen Format: {chosen_format}")

    def custom_remove_stop_words(self, sentence):
        stop_words = set(['rt', 'twitter', 'retweet', 'https'])
        stop_words.update(stopwords.words('english'))
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words and not word.startswith('@')]
        return ' '.join(filtered_words)

if __name__ == "__main__":
    root = tk.Tk()
    main_page = MainPage(root)
    root.mainloop()