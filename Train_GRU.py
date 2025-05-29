
# import time
# import pickle
# import tensorflow as tf
# import pandas as pd
# import tqdm
# import numpy as np
# import pandas as pd
# import re
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Dense, Dropout, GRU
# from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.stem import PorterStemmer, LancasterStemmer
# from nltk.stem.snowball import SnowballStemmer
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
# import string
# from string import punctuation
# import nltk
# import os
# from keras.models import load_model


# SEQUENCE_LENGTH = 100 # the length of all sequences (number of words per sample)
# EMBEDDING_SIZE = 100  # Using 100-Dimensional GloVe embedding vectors
# TEST_SIZE = 0.25 # ratio of testing set
# max_features = 100
# BATCH_SIZE = 64

# maxlen = 500
# EPOCHS = 10 # number of epochs

# def load_data():
#     """
#     Loads Youtube comments Collection dataset
#     """
#     data = pd.read_csv("comments.csv",encoding='latin-1')

#     data['Comment'] =  data['Comment'].apply(lambda text: text_processing(text))

#     texts = data['Comment'].values

#     labels=data['Sentiment'].values


#     return texts, labels


# def text_processing(text):
#     stop_words = stopwords.words('english')
#     porter_stemmer = PorterStemmer()
#     lancaster_stemmer = LancasterStemmer()
#     snowball_stemer = SnowballStemmer(language="english")
#     lzr = WordNetLemmatizer()

#     if isinstance(text, str):
#         # convert text into lowercase
#         text = text.lower()

#         # remove new line characters in text
#         text = re.sub(r'\n', ' ', text)

#         # remove punctuations from text
#         text = re.sub('[%s]' % re.escape(punctuation), "", text)

#         # remove references and hashtags from text
#         text = re.sub("^a-zA-Z0-9$,.", "", text)

#         # remove multiple spaces from text
#         text = re.sub(r'\s+', ' ', text, flags=re.I)

#         # remove special characters from text
#         text = re.sub(r'\W', ' ', text)

#         text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

#         # stemming using porter stemmer from nltk package - msh a7sn 7aga - momken: lancaster, snowball
#         # text=' '.join([porter_stemmer.stem(word) for word in word_tokenize(text)])
#         # text=' '.join([lancaster_stemmer.stem(word) for word in word_tokenize(text)])
#         # text=' '.join([snowball_stemer.stem(word) for word in word_tokenize(text)])

#         # lemmatizer using WordNetLemmatizer from nltk package
#         text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])
#     else:
#         text = ''


#     return text

# def gru_model_training():
#     print("loading data")
#     X, y = load_data()

#     # Text tokenization
#     # vectorizing text, turning each text into sequence of integers
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(X)
#     # lets dump it to a file, so we can use it in testing
#     pickle.dump(tokenizer, open("gru_tokenizer.pickle", "wb"))
#     # convert to sequence of integers
#     X = tokenizer.texts_to_sequences(X)

#     # convert to numpy arrays
#     X = np.array(X)
#     y = np.array(y)
#     # pad sequences at the beginning of each sequence with 0's
#     # for example if SEQUENCE_LENGTH=4:
#     # [[5, 3, 2], [5, 1, 2, 3], [3, 4]]
#     # will be transformed to:
#     # [[0, 5, 3, 2], [5, 1, 2, 3], [0, 0, 3, 4]]
#     X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

#     # One Hot encoding labels
#     # [spam, ham, spam, ham, ham] will be converted to:
#     # [1, 0, 1, 0, 1] and then to:
#     # [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]

#     #y = [label2int[label] for label in y]
#     y = to_categorical(y)

#     # split and shuffle


#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
#     accuracy=0; precision=0; recall=0; fscore=0

#     if os.path.exists("gru_model.h5"):
#         model_path = 'gru_model.h5'
#         model = load_model(model_path)
#         y_pred = model.predict(X_test)
#         y_pred = np.argmax(y_pred, axis=1)
#         y_test = np.argmax(y_test, axis=1)

#         accuracy = accuracy_score(y_test, y_pred) * 100
#         precision = precision_score(y_test, y_pred, average="macro") * 100
#         recall = recall_score(y_test, y_pred, average="macro") * 100
#         fscore = f1_score(y_test, y_pred, average="macro") * 100
#         print("GRU=", accuracy, precision, recall, fscore)

#     else:

#         embedding_matrix = get_embedding_vectors(tokenizer)

#         # Build the model
#         model = Sequential()
#         model.add(Embedding(len(tokenizer.word_index) + 1,
#                             EMBEDDING_SIZE,
#                             weights=[embedding_matrix],
#                             trainable=False,
#                             input_length=SEQUENCE_LENGTH))
#         model.add(GRU(200))
#         model.add(Dense(3, activation='softmax'))

#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#         model.fit(X_train, y_train, epochs=20, verbose=1, validation_data=(X_test, y_test), batch_size=64)
#         # print("saving")
#         model.save('gru_model.h5')

#         y_test = np.argmax(y_test, axis=1)
#         y_pred = np.argmax(model.predict(X_test), axis=1)

#         accuracy = accuracy_score(y_test, y_pred) * 100

#         precision = precision_score(y_test, y_pred, average="macro") * 100

#         recall = recall_score(y_test, y_pred, average="macro") * 100

#         fscore = f1_score(y_test, y_pred, average="macro") * 100

        



#     return accuracy, precision, recall, fscore


# def get_embedding_vectors(tokenizer, dim=100):
#     embedding_index = {}
#     with open(f"glove.6B.{dim}d.txt", encoding='utf8') as f:
#         for line in tqdm.tqdm(f, "Reading GloVe"):
#             values = line.split()
#             word = values[0]
#             vectors = np.asarray(values[1:], dtype='float32')
#             embedding_index[word] = vectors

#     word_index = tokenizer.word_index
#     embedding_matrix = np.zeros((len(word_index) + 1, dim))
#     for word, i in word_index.items():
#         embedding_vector = embedding_index.get(word)
#         if embedding_vector is not None:
#             # words not found will be 0s
#             embedding_matrix[i] = embedding_vector

#     return embedding_matrix


# if __name__ == '__main__':
#     gru_model_training()

import time
import pickle
import tensorflow as tf
import pandas as pd
import tqdm
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Dense, GRU
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation

SEQUENCE_LENGTH = 100
EMBEDDING_SIZE = 100
TEST_SIZE = 0.25
BATCH_SIZE = 64
EPOCHS = 10

def text_processing(text):
    stop_words = stopwords.words('english')
    lzr = WordNetLemmatizer()
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\n', ' ', text)
        text = re.sub('[%s]' % re.escape(punctuation), "", text)
        text = re.sub("^a-zA-Z0-9$,.", "", text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        text = re.sub(r'\W', ' ', text)
        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
        text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])
    else:
        text = ''
    return text

def load_data():
    data = pd.read_csv("comments.csv", encoding='latin-1')
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    if data['Sentiment'].dtype == 'object':
        data['Sentiment'] = data['Sentiment'].map(label_map)
    data = data.dropna(subset=["Sentiment"])
    data['Sentiment'] = data['Sentiment'].astype(int)
    data['Comment'] = data['Comment'].apply(text_processing)
    return data['Comment'].values, data['Sentiment'].values

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
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def gru_model_training():
    print("🔹 Loading Data...")
    X, y = load_data()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    pickle.dump(tokenizer, open("gru_tokenizer.pickle", "wb"))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)  # ✅ FIXED padding before np.array
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

    # ⚖️ Class weights
    original_labels = np.argmax(y, axis=1)
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(original_labels), y=original_labels)
    class_weights = dict(enumerate(class_weights))
    print("⚖️ Class Weights:", class_weights)

    if os.path.exists("gru_model.h5"):
        print("📂 Loading Pre-Trained Model...")
        model = load_model("gru_model.h5")
    else:
        print("🚀 Training New GRU Model...")
        embedding_matrix = get_embedding_vectors(tokenizer)

        model = Sequential([
            Embedding(len(tokenizer.word_index) + 1, EMBEDDING_SIZE,
                      weights=[embedding_matrix], trainable=False, input_length=SEQUENCE_LENGTH),
            GRU(200),
            Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test),
                  batch_size=BATCH_SIZE, class_weight=class_weights, verbose=1)

        model.save('gru_model.h5')

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average="macro") * 100
    recall = recall_score(y_test, y_pred, average="macro") * 100
    fscore = f1_score(y_test, y_pred, average="macro") * 100

    print(f"🎯 GRU Model Metrics: Accuracy={accuracy:.2f}% Precision={precision:.2f}% Recall={recall:.2f}% F1-score={fscore:.2f}%")
    return accuracy, precision, recall, fscore

if __name__ == '__main__':
    gru_model_training()
