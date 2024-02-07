import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional
import pickle
import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# nltk.download('stopwords')

def custom_remove_stop_words(sentence):
    stop_words = set(['rt', 'twitter', 'retweet', 'https'])
    stop_words.update(stopwords.words('english'))
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words and not word.startswith('@')]
    return ' '.join(filtered_words)

output_df = pd.read_csv('sentence_database_3.csv')
output_df['sentence'] = output_df['sentence'].apply(custom_remove_stop_words)

output_df['text_length'] = output_df['sentence'].apply(len)
sentence_label = output_df['label'].values

# ------------------------ word cloud --------------------------------------- #
# offensive_msg = output_df[output_df.label ==1]
# non_offensive_msg = output_df[output_df.label==0]

# offensive_msg_text = " ".join(offensive_msg.sentence.to_numpy().tolist())
# non_offensive_msg_text = " ".join(non_offensive_msg.sentence.to_numpy().tolist())


# offensive_msg_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Blues').generate(offensive_msg_text)
# plt.figure(figsize=(10,5))
# plt.imshow(offensive_msg_cloud, interpolation='bilinear')
# plt.axis('off') # turn off axis
# plt.show()

# non_offensive_msg_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Pastel1').generate(non_offensive_msg_text)
# plt.figure(figsize=(10,5))
# plt.imshow(non_offensive_msg_cloud, interpolation='bilinear')
# plt.axis('off') # turn off axis
# plt.show()
# ------------------------ word cloud --------------------------------------- #

x_train, x_test, y_train, y_test = train_test_split(output_df['sentence'], sentence_label, test_size=0.2, random_state=434)

tokenizer = Tokenizer(num_words=4000, char_level=False, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

word_index_path = "word_index.pkl"
with open(word_index_path, 'wb') as word_index_file:
    pickle.dump(tokenizer.word_index, word_index_file)

training_sequences = tokenizer.texts_to_sequences(x_train)
training_padded = pad_sequences(training_sequences, maxlen=100, padding='post', truncating='post')

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences, maxlen=100, padding='post', truncating='post')

# Define the sentence-level model
vocab_size = 4000 
embedding_dim = 64
drop_value = 0.2

# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, input_length=100))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(drop_value))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the sentence-level model
# num_epochs = 30
# early_stop = EarlyStopping(monitor='val_loss', patience=3)
# history = model.fit(training_padded,
#                     y_train,
#                     epochs=num_epochs, 
#                     validation_data=(testing_padded, y_test),
#                     callbacks=[early_stop],
#                     verbose=2)
# model.evaluate(testing_padded, y_test)

# metrics = pd.DataFrame(history.history)
# # Rename column
# metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy', 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
# def plot_graphs1(var1, var2, string):
#     metrics[[var1, var2]].plot()
#     plt.title('Training and Validation ' + string)
#     plt.xlabel ('Number of epochs')
#     plt.ylabel(string)
#     plt.legend([var1, var2])

# plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
# plt.show()

# plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
# plt.show()

# model.save("sentence_model_dense.h5")

# # -------------------------------------------------------------------#

n_lstm = 128
drop_lstm = 0.2
# # Define LSTM Model
# model1 = Sequential()
# model1.add(Embedding(vocab_size, embedding_dim, input_length=100))
# model1.add(SpatialDropout1D(drop_lstm))
# model1.add(LSTM(n_lstm, return_sequences=False))
# model1.add(Dropout(drop_lstm))
# model1.add(Dense(1, activation='sigmoid'))

# model1.compile(loss = 'binary_crossentropy',
#                optimizer = 'adam',
#                metrics = ['accuracy'])

# num_epochs = 30
# early_stop = EarlyStopping(monitor='val_loss', patience=2)
# history = model1.fit(training_padded,
#                      y_train,
#                      epochs=num_epochs,
#                      validation_data=(testing_padded, y_test),
#                      callbacks =[early_stop],
#                      verbose=2)

# # Create a dataframe
# metrics = pd.DataFrame(history.history)
# # Rename column
# metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
#                          'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
# def plot_graphs1(var1, var2, string):
#     metrics[[var1, var2]].plot()
#     plt.title('LSTM Model: Training and Validation ' + string)
#     plt.xlabel ('Number of epochs')
#     plt.ylabel(string)
#     plt.legend([var1, var2])
# plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
# plt.show()
# plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
# plt.show()

# model1.save("sentence_model_lstm.h5")

# -------------------------------------------------------------------#

model2 = Sequential()
model2.add(Embedding(vocab_size,
                     embedding_dim,
                     input_length = 100))
model2.add(Bidirectional(LSTM(n_lstm,
                              return_sequences = False)))
model2.add(Dropout(drop_lstm))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss = 'binary_crossentropy',
               optimizer = 'adam',
               metrics=['accuracy'])
num_epochs = 30
early_stop = EarlyStopping(monitor = 'val_loss',
                           patience = 2)
history = model2.fit(training_padded,
                     y_train,
                     epochs = num_epochs,
                     validation_data = (testing_padded, y_test),
                     callbacks = [early_stop],
                     verbose = 2)

# metrics = pd.DataFrame(history.history)
# # Rename column
# metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
#                          'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
# def plot_graphs1(var1, var2, string):
#     metrics[[var1, var2]].plot()
#     plt.title('BiLSTM Model: Training and Validation ' + string)
#     plt.xlabel ('Number of epochs')
#     plt.ylabel(string)
#     plt.legend([var1, var2])
# # Plot
# plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
# plt.show()
# plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
# plt.show()

# print(f"Dense model loss and accuracy: {model.evaluate(testing_padded, y_test)} " )
# print(f"LSTM model loss and accuracy: {model1.evaluate(testing_padded, y_test)} " )
# print(f"Bi-LSTM model loss and accuracy: {model2.evaluate(testing_padded, y_test)} " )

# model2.save("sentence_model_bi_lstm.h5")

# model.evaluate(testing_padded, y_test)

# input_sentence = 'What is your sex?'
# input_sequence = tokenizer.texts_to_sequences([input_sentence])
# input_padded = pad_sequences(input_sequence, maxlen=50, padding='post', truncating='post')

# # Make predictions
# prediction = model.predict(input_padded)

# # Print the prediction
# print(prediction)

# # Filter sentences with predicted probability > 80%
# threshold = 0.8
# offensive_sentences_indices = [i for i, prob in enumerate(model2.predict(testing_padded)) if prob > threshold]
# offensive_sentences = x_test.iloc[offensive_sentences_indices]

# def predict_offensiveness(sentence):
#     sequence = tokenizer.texts_to_sequences([sentence])
#     padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    
#     prediction = model2.predict(padded_sequence)
    
#     return prediction

# def censor_offensive_words(sentence, threshold=0.8):
#     words = sentence.split()
#     censored_words = []

#     for word in words:
#         word_prediction = predict_offensiveness(word.lower())
#         print(word)
#         print(word_prediction)

#         if word_prediction > threshold:
#             censored_words.append("*" * len(word))
#         else:
#             censored_words.append(word)

#     censored_sentence = ' '.join(censored_words)
    
#     return censored_sentence

# # Example usage:
# example_sentence = "mom loves me the most"
# example_sentence = custom_remove_stop_words(example_sentence)
# overall_prediction = predict_offensiveness(example_sentence.lower())
# print(overall_prediction)

# if overall_prediction > 0.8:
#     censored_sentence = censor_offensive_words(example_sentence)
#     print("Original Sentence:", example_sentence)
#     print("Censored Sentence:", censored_sentence)
# else:
#     print("The sentence is not offensive.")