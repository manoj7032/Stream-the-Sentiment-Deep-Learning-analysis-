# 🎯 Stream the Sentiment - Deep Learning analysis (Flask + BiLSTM)

This project is a full-stack web application that performs **sentiment analysis on YouTube video comments** using deep learning models such as **Bi-LSTM, GRU, LSTM, and RNN**. It allows users to **analyze sentiment**, **view results**, and **compare models**, all through an interactive Flask-based dashboard.

---

## Features

- Search any YouTube video via video ID
- Extract and clean comments using YouTube Data API
- Perform sentiment classification (Positive, Neutral, Negative)
- Visualize sentiment distribution using pie charts
- Supports multiple deep learning models:
  - Bi-LSTM
  - LSTM
  - GRU
  - RNN
- Compare model performance metrics: Accuracy, Precision, Recall, F1 Score
- Login/Register system for users
- Dataset viewing and analysis panel
- History tracking per user

---

## 🛠 Technologies Used

- **Frontend**: HTML, CSS, Bootstrap (via templates)
- **Backend**: Python, Flask
- **Database**: MySQL
- **Machine Learning**: TensorFlow / Keras
- **Visualization**: Matplotlib, Seaborn
- **APIs**: YouTube Data API v3

---
Download the following Glove and H5 files and place it in the root project folder. 
Download link - https://drive.google.com/drive/folders/1jrn5CiZzkF1yWtmlD1Yaxr0b6nroM0Dy?usp=sharing

## Folder Structure

```plaintext
|-- index.py               # Main Flask app
|-- sentiment.py           # Predicts sentiment using trained Bi-LSTM
|-- Train_Bi_LSTM.py       # Trains Bi-LSTM model
|-- Train_GRU.py           # Trains GRU model
|-- Train_LSTM.py          # Trains LSTM model
|-- Train_RNN.py           # Trains RNN model
|-- test_sentiment.py      # Script to test sentiment inference
|-- dataset.py             # Dataset visualization logic
|-- youtube.py             # YouTube comment & metadata fetching
|-- DBConfig.py            # MySQL database configuration
|-- comments.csv           # Dataset (already preprocessed)
|-- templates/             # HTML templates (login, dashboard, etc.)
|-- static/                # CSS, images, JS (if any)


