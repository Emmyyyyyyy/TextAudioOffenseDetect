import re
import csv
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

def load_offensive_words_from_csv(csv_file):
    offensive_data = {'word': [], 'label': []}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['word'] and row['label']:
                offensive_data['word'].append(row['word'])
                offensive_data['label'].append(row['label'])
    return offensive_data

def load_word_index_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as word_index_file:
        word_index = pickle.load(word_index_file)
    return word_index

def preprocess_word(word):
    # Remove common suffixes
    suffixes = ['s', 'ing', 'ed']
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
    return word.lower()

sentence_model = load_model('sentence_model_bi_lstm.h5')

def predict_offensiveness(sentence):
    global sentence_model

    word_index_path = "word_index.pkl"
    with open(word_index_path, 'rb') as word_index_file:
        word_index = pickle.load(word_index_file)

    tokenizer = Tokenizer(num_words=4000, char_level=False, oov_token="<OOV>")
    tokenizer.word_index = word_index

    input_sequence = tokenizer.texts_to_sequences([sentence])
    input_padded = pad_sequences(input_sequence, maxlen=100, padding='post', truncating='post')

    prediction = sentence_model.predict(input_padded)
    print(prediction)
    
    return prediction

def detect_offensive_word(sentence, timestamp):
    offensive_data = load_offensive_words_from_csv('word_database_3.csv')
    offensive_words_found = []

    for word in offensive_data['word']:
        pattern = r'\b(?:' + re.escape(word) + r'|' + '|'.join(re.escape(word) + form for form in ["", "s", "ed", "ing"]) + r')\b'
        regex_pattern = re.compile(pattern, flags=re.IGNORECASE)

        if not regex_pattern.search(sentence):
            continue

        index = offensive_data['word'].index(word)
        label = offensive_data['label'][index]

        if label == '1':
            word_positions = [(item['start'], item['end']) for item in timestamp if preprocess_word(item['word']) == preprocess_word(word)]
            if word_positions:
                for start, end in word_positions:
                    offensive_words_found.append({"word": word, "start": start, "end": end})
            else:
                offensive_words_found.append({"word": word, "start": 0.00, "end": 0.0})
    print(offensive_words_found)
    return offensive_words_found

def censor_offensive_words(sentence, timestamp, threshold=0.8):
    words = sentence.split()
    censored_words = set()

    for word in words:
        print(word)
        offensive_word_info = detect_offensive_word(word, timestamp)
        if offensive_word_info:
            censored_words.update((entry['word'], entry['start'], entry['end']) for entry in offensive_word_info)
        else:
            word_prediction = predict_offensiveness(word.lower())
            if word_prediction > threshold:
                word_positions = [
                    (item['word'], item['start'], item['end'])
                    for item in timestamp
                    if preprocess_word(item['word']) == preprocess_word(word)
                ]

                if word_positions:
                    censored_words.update(word_positions)
                else:
                    censored_words.add((word, 0.00, 0.0))

    censored_words_list = [{"word": word, "start": start, "end": end} for word, start, end in censored_words]
    print(censored_words_list)
    return censored_words_list


word_index_path = "word_index.pkl"

# timestamp_data = [{'word': 'put', 'start': 0.72, 'end': 0.89}, {'word': 'your', 'start': 0.89, 'end': 1.05}, {'word': 'Dick', 'start': 1.05, 'end': 1.33}, {'word': 'on', 'start': 1.33, 'end': 1.51}, {'word': 'my', 'start': 1.51, 'end': 1.68}, {'word': 'pussy', 'start': 1.68, 'end': 2.19},{'word': 'fuck', 'start': 2.19, 'end': 2.45}, {'word': 'fucking', 'start': 2.45, 'end': 2.83}, {'word': 'have', 'start': 2.83, 'end': 3.01}, {'word': 'K.', 'start': 3.75, 'end': 3.98}, {'word': 'Y.', 'start': 3.98, 'end': 4.3}]

# test_sentence = 'put your Dick on my pussy fucking have K. Y. '

# censored_words = censor_offensive_words(test_sentence, timestamp_data)