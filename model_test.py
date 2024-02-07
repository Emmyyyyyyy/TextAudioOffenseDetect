import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle

def load_word_index(word_index_path):
    with open(word_index_path, 'rb') as word_index_file:
        word_index = pickle.load(word_index_file)
    return word_index

# Load model and word index outside the function
sentence_model = load_model('sentence_model_bi_lstm.h5')
word_index_path = "word_index.pkl"
word_index = load_word_index(word_index_path)

def predict_label(sentence):
    global sentence_model, word_index
    input_text_without_stopwords = custom_remove_stop_words(sentence)

    tokenizer = Tokenizer(num_words=4000, char_level=False, oov_token="<OOV>")
    tokenizer.word_index = word_index

    input_sequence = tokenizer.texts_to_sequences([input_text_without_stopwords])
    input_padded = pad_sequences(input_sequence, maxlen=100, padding='post', truncating='post')

    prediction = sentence_model.predict(input_padded)
    print(sentence)
    print(prediction)
    return 1 if prediction[0][0] > 0.8 else 0

def custom_remove_stop_words(sentence):
    if isinstance(sentence, str):
        stop_words = set(['rt', 'twitter', 'retweet', 'https'])
        stop_words.update(stopwords.words('english'))
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words and not word.startswith('@')]
        return ' '.join(filtered_words)
    else:
        # Handle non-string values (e.g., NaN) as needed
        return ''

df = pd.read_csv('./sentence_database_3.csv')

df['predicted_label'] = df['sentence'].apply(predict_label).astype(int)

correlation_table = pd.crosstab(df['predicted_label'], df['label'], margins=True, margins_name='Total')

accuracy = (df['predicted_label'] == df['label']).mean()
print(f"Accuracy: {accuracy}")

print("\nCorrelation Table:")
print(correlation_table)
