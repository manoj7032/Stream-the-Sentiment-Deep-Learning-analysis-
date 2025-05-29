
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
import numpy as np
import pickle
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

def text_processing(text):
    stop_words = stopwords.words('english')
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

def get_predictions(text):
    model_path = 'bilstm_model.h5'
    model = load_model(model_path)
    model_path2 = 'bilstm_tokenizer.pickle'

    with open(model_path2, 'rb') as f:
        tokenizer = pickle.load(f)
    X = tokenizer.texts_to_sequences([text])

    # convert to numpy arrays
    sequence = np.array(X)

    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=100)
    # get the prediction
    prediction = model.predict(sequence)[0]
    print(prediction)
    print(np.argmax(prediction))
    # one-hot encoded vector, revert using np.argmax
    return np.argmax(prediction)

if __name__ == '__main__':

    text = "I hate this fucking video"
    text=text_processing(text)
    print(get_predictions(text))