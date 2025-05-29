import time
import pickle
import tensorflow as tf
import pandas as pd
import tqdm
import numpy as np
import pandas as pd
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
#from tensorflow.keras.layers import Embedding, Dropout, Dense
from tensorflow.keras.models import Sequential
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import LSTM, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten,BatchNormalization

from nltk.corpus import stopwords

from string import punctuation
#from tensorflow.keras.metrics import Recall, Precision

from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from keras.models import load_model
from tensorflow.keras.layers import Conv1D,LSTM, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation
import nltk
import re

SEQUENCE_LENGTH = 100 # the length of all sequences (number of words per sample)
EMBEDDING_SIZE = 100  # Using 100-Dimensional GloVe embedding vectors
TEST_SIZE = 0.25 # ratio of testing set

BATCH_SIZE = 64
EPOCHS = 10 # number of epochs

def load_data():
    """
    Loads Youtube comments Collection dataset
    """
    data = pd.read_csv("comments.csv",encoding='latin-1')

    data['Comment'] =  data['Comment'].apply(lambda text: text_processing(text))

    texts = data['Comment'].values

    labels=data['Sentiment'].values

    return texts, labels

def text_processing(text):
    # nltk.download('stopwords')
    stop_words = stopwords.words('english')
    # stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    snowball_stemer = SnowballStemmer(language="english")
    lzr = WordNetLemmatizer()

    if isinstance(text, str):
        # convert text into lowercase
        text = text.lower()

        # remove new line characters in text
        text = re.sub(r'\n', ' ', text)

        # remove punctuations from text
        text = re.sub('[%s]' % re.escape(punctuation), "", text)

        # remove references and hashtags from text
        text = re.sub("^a-zA-Z0-9$,.", "", text)

        # remove multiple spaces from text
        text = re.sub(r'\s+', ' ', text, flags=re.I)

        # remove special characters from text
        text = re.sub(r'\W', ' ', text)

        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

        # stemming using porter stemmer from nltk package - msh a7sn 7aga - momken: lancaster, snowball
        # text=' '.join([porter_stemmer.stem(word) for word in word_tokenize(text)])
        # text=' '.join([lancaster_stemmer.stem(word) for word in word_tokenize(text)])
        # text=' '.join([snowball_stemer.stem(word) for word in word_tokenize(text)])

        # lemmatizer using WordNetLemmatizer from nltk package
        text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])
    else:
        text = ''

    return text

# def lstm_model_training():
    #print("loading data")
    #X, y = load_data()

    # Text tokenization
    # vectorizing text, turning each text into sequence of integers
    #tokenizer = Tokenizer()
    #tokenizer.fit_on_texts(X)
    # lets dump it to a file, so we can use it in testing
    #pickle.dump(tokenizer, open("lstm_tokenizer.pickle", "wb"))
    # convert to sequence of integers
    #X = tokenizer.texts_to_sequences(X)
    # X = pad_sequences(X, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')
    # convert to numpy arrays
    #X = np.array(X)
    #y = np.array(y)
    # pad sequences at the beginning of each sequence with 0's
    # for example if SEQUENCE_LENGTH=4:
    # [[5, 3, 2], [5, 1, 2, 3], [3, 4]]
    # will be transformed to:
    # [[0, 5, 3, 2], [5, 1, 2, 3], [0, 0, 3, 4]]
    #X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)
    # X = pad_sequences(X, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')

    # One Hot encoding labels
    # [spam, ham, spam, ham, ham] will be converted to:
    # [1, 0, 1, 0, 1] and then to:
    # [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]

    #y = [label2int[label] for label in y]
    #y = to_categorical(y)

    # split and shuffle
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    # print our data shapes
    '''print("X_train.shape:", X_train.shape)
    print("X_test.shape:", X_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)'''

    #accuracy=0; precision=0; recall=0; fscore=0

    #if os.path.exists("lstm_model.h5"):
     #   model_path = 'lstm_model.h5'
      #  model = load_model(model_path)
       # y_pred = model.predict(X_test)
        #y_pred = np.argmax(y_pred, axis=1)
        #y_test = np.argmax(y_test, axis=1)

        #accuracy = accuracy_score(y_test, y_pred) * 100
        #precision = precision_score(y_test, y_pred, average="macro") * 100
        #recall = recall_score(y_test, y_pred, average="macro") * 100
        #fscore = f1_score(y_test, y_pred, average="macro") * 100
        #print("LSTM=", accuracy, precision, recall, fscore)

    #else:

        # print("EMD Matrix")
     #   embedding_matrix = get_embedding_vectors(tokenizer)
      #  print("Starting...")
       # model = Sequential()
        #model.add(Embedding(len(tokenizer.word_index) + 1,
         #                   EMBEDDING_SIZE,
          #                  weights=[embedding_matrix],
           #                 trainable=False,
            #                input_length=SEQUENCE_LENGTH))

        #model.add(LSTM(32, return_sequences=True))
        #model.add(BatchNormalization())

        #model.add(LSTM(64))
        #model.add(BatchNormalization())

        #model.add(Dense(64, activation='relu'))
        #model.add(Dense(3, activation="softmax"))
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        #model.fit(X, y, epochs=20, verbose=1, validation_data=(X_test, y_test), batch_size=64)
        # print("saving")
        #model.save('lstm_model.h5')
        # model.summary()

        #y_test = np.argmax(y_test, axis=1)
        #y_pred = np.argmax(model.predict(X_test), axis=1)

        #acc = accuracy_score(y_test, y_pred) * 100

        #precsn = precision_score(y_test, y_pred, average="macro") * 100

        #recall = recall_score(y_test, y_pred, average="macro") * 100

        #f1score = f1_score(y_test, y_pred, average="macro") * 100

        #print("LSTM=", acc, precsn, recall, f1score)

    #return accuracy, precision, recall, fscore

def lstm_model_training():
    """Trains LSTM model on YouTube comments dataset"""
    print("🔹 Loading Data...")
    X, y = load_data()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    pickle.dump(tokenizer, open("lstm_tokenizer.pickle", "wb"))

    # Convert text to sequences of integers
    X = tokenizer.texts_to_sequences(X)

    # 🚀 Ensure all elements in X are valid lists (fixing empty sequences)
    X = [seq if isinstance(seq, list) else [] for seq in X]

    # 🚀 Pad sequences to ensure uniform shape
    X = pad_sequences(X, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')

    # 🚀 Convert to NumPy array after padding
    X = np.array(X)
    y = np.array(y)

    # 🚀 Debugging: Print first few samples
    print("✅ First 5 padded sequences:", X[:5])
    print("✅ Shape after padding:", X.shape)

    # One Hot encoding labels
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    print("🚀 Training New LSTM Model...")
    embedding_matrix = get_embedding_vectors(tokenizer)

    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, EMBEDDING_SIZE, weights=[embedding_matrix], trainable=False),
        LSTM(32, return_sequences=True),
        BatchNormalization(),
        LSTM(64),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dense(3, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test), batch_size=BATCH_SIZE)

    model.save("lstm_model.h5")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average="macro") * 100
    recall = recall_score(y_test, y_pred, average="macro") * 100
    fscore = f1_score(y_test, y_pred, average="macro") * 100

    print("🎯 LSTM Model Metrics:", accuracy, precision, recall, fscore)

    return accuracy, precision, recall, fscore


def get_embedding_vectors(tokenizer, dim=100):
    embedding_index = {}
    with open(f"glove.6B.{dim}d.txt", encoding='utf8') as f:
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

if __name__ == '__main__':
    lstm_model_training()
