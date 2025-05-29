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

def get_predictions(text_list):
    model_path = 'bilstm_model.h5'
    model = load_model(model_path)
    model_path2 = 'bilstm_tokenizer.pickle'

    with open(model_path2, 'rb') as f:
        tokenizer = pickle.load(f)

 
    predictions = []
    for text in text_list:
        processed_text = text_processing(text)
        X = tokenizer.texts_to_sequences([processed_text])

       
        sequence = pad_sequences(np.array(X), maxlen=100)

       
        prediction = model.predict(sequence)[0]
        predictions.append(np.argmax(prediction))

    return predictions

if __name__ == '__main__':
   
    texts = [
        "I hate this video",
        "I love the way this looks",
        "This is an average product",
        "The service is fantastic"
    ]

    predictions = get_predictions(texts)
    print(type(predictions))
    for i, text in enumerate(texts):
        print(f"Text: {text} - Sentiment: {predictions[i]}")
