# import time
# import pickle
# import tensorflow as tf
# import pandas as pd
# import tqdm
# import numpy as np
# import pandas as pd
# import re
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from sklearn.model_selection import train_test_split
# #from tensorflow.keras.layers import Embedding, Dropout, Dense
# from tensorflow.keras.models import Sequential
# from nltk.tokenize import word_tokenize
# from tensorflow.keras.layers import LSTM, Bidirectional,GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten,BatchNormalization
# from keras.models import load_model
# from nltk.corpus import stopwords
# import os
# from string import punctuation
# #from tensorflow.keras.metrics import Recall, Precision

# from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

# from tensorflow.keras.layers import Conv1D,LSTM, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.stem import PorterStemmer, LancasterStemmer
# from nltk.stem.snowball import SnowballStemmer
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
# import string
# from string import punctuation
# import nltk
# import re

# SEQUENCE_LENGTH = 100 
# EMBEDDING_SIZE = 100  
# TEST_SIZE = 0.25 

# BATCH_SIZE = 64
# EPOCHS = 10 
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
        
#         text = text.lower()

        
#         text = re.sub(r'\n', ' ', text)

        
#         text = re.sub('[%s]' % re.escape(punctuation), "", text)

        
#         text = re.sub("^a-zA-Z0-9$,.", "", text)

       
#         text = re.sub(r'\s+', ' ', text, flags=re.I)

        
#         text = re.sub(r'\W', ' ', text)

#         text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

      
#         text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])
#     else:
#         text = ''


#     return text

# def bilstm_model_training():
#     print("loading data")
#     X, y = load_data()

   
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(X)
    
#     pickle.dump(tokenizer, open("bilstm_tokenizer.pickle", "wb"))
   
#     X = tokenizer.texts_to_sequences(X)

   

    
#     X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

#     X = np.array(X)
#     y = np.array(y)

    
#     y = to_categorical(y)

   
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
   
#     '''print("X_train.shape:", X_train.shape)
#     print("X_test.shape:", X_test.shape)
#     print("y_train.shape:", y_train.shape)
#     print("y_test.shape:", y_test.shape)'''

#     accuracy=0; precision=0; recall=0; fscore=0
#     if os.path.exists("bilstm_model.h5"):
#         model_path = 'bilstm_model.h5'
#         model = load_model(model_path)
#         y_pred = model.predict(X_test)
#         y_pred = np.argmax(y_pred, axis=1)
#         y_test = np.argmax(y_test, axis=1)

#         accuracy = accuracy_score(y_test, y_pred) * 100
#         precision = precision_score(y_test, y_pred, average="macro") * 100
#         recall = recall_score(y_test, y_pred, average="macro") * 100
#         fscore = f1_score(y_test, y_pred, average="macro") * 100
#         print("BiLSTM=", accuracy, precision, recall, fscore)
#     else:
       
#         embedding_matrix = get_embedding_vectors(tokenizer)
#         print("Starting...")
#         model = Sequential()
#         model.add(Embedding(len(tokenizer.word_index) + 1,
#                             EMBEDDING_SIZE,
#                             weights=[embedding_matrix],
#                             trainable=False,
#                             input_length=SEQUENCE_LENGTH))

#         model.add(Bidirectional(LSTM(32, return_sequences=True)))
#         model.add(BatchNormalization())

#         model.add(Bidirectional(LSTM(64)))
#         model.add(BatchNormalization())

#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(3, activation="softmax"))
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#         model.fit(X, y, epochs=20, verbose=1, validation_data=(X_test, y_test), batch_size=64)
        
#         model.save('bilstm_model.h5')
        

#         y_test = np.argmax(y_test, axis=1)
#         y_pred = np.argmax(model.predict(X_test), axis=1)

#         acc = accuracy_score(y_test, y_pred) * 100

#         precsn = precision_score(y_test, y_pred, average="macro") * 100

#         recall = recall_score(y_test, y_pred, average="macro") * 100

#         f1score = f1_score(y_test, y_pred, average="macro") * 100

#         print("BiLSTM=", acc, precsn, recall, f1score)



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
            
#             embedding_matrix[i] = embedding_vector

#     return embedding_matrix



# if __name__ == '__main__':
#     bilstm_model_training()

import time
import pickle
import tensorflow as tf
import pandas as pd
import tqdm
import numpy as np
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Bidirectional, BatchNormalization, Dense, Embedding
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer, SnowballStemmer
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

    # Map text labels if present
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    if data['Sentiment'].dtype == 'object':
        data['Sentiment'] = data['Sentiment'].map(label_map)

    data = data.dropna(subset=["Sentiment"])
    data['Sentiment'] = data['Sentiment'].astype(int)

    data['Comment'] = data['Comment'].apply(text_processing)
    texts = data['Comment'].values
    labels = data['Sentiment'].values
    return texts, labels

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

def bilstm_model_training():
    print("🔹 Loading Data...")
    X, y = load_data()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    pickle.dump(tokenizer, open("bilstm_tokenizer.pickle", "wb"))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)  # ✅ FIXED

    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

    # ⚖️ Compute class weights
    original_labels = np.argmax(y, axis=1)
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(original_labels), y=original_labels)
    class_weights = dict(enumerate(class_weights))
    print("⚖️ Class Weights:", class_weights)

    accuracy = precision = recall = fscore = 0

    if os.path.exists("bilstm_model.h5"):
        print("📂 Loading Pre-Trained Model...")
        model = load_model("bilstm_model.h5")
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
    else:
        print("🚀 Training BiLSTM Model...")
        embedding_matrix = get_embedding_vectors(tokenizer)

        model = Sequential([
            Embedding(len(tokenizer.word_index) + 1, EMBEDDING_SIZE,
                      weights=[embedding_matrix], trainable=False, input_length=SEQUENCE_LENGTH),
            Bidirectional(LSTM(32, return_sequences=True)),
            BatchNormalization(),
            Bidirectional(LSTM(64)),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dense(3, activation="softmax")
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.fit(X, y, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test),
                  batch_size=BATCH_SIZE, class_weight=class_weights)
        model.save('bilstm_model.h5')
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average="macro") * 100
    recall = recall_score(y_test, y_pred, average="macro") * 100
    fscore = f1_score(y_test, y_pred, average="macro") * 100

    print(f"🎯 BiLSTM Model Metrics: Accuracy={accuracy:.2f}% Precision={precision:.2f}% Recall={recall:.2f}% F1-score={fscore:.2f}%")
    return accuracy, precision, recall, fscore

if __name__ == '__main__':
    bilstm_model_training()
