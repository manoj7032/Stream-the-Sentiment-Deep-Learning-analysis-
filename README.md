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

## Technologies Used

- **Frontend**: HTML, CSS, Bootstrap (via templates)
- **Backend**: Python, Flask
- **Database**: MySQL
- **Machine Learning**: TensorFlow / Keras
- **Visualization**: Matplotlib, Seaborn
- **APIs**: YouTube Data API v3

---
Download the following Glove and H5 files and place it in the root project folder. 
Download link - https://drive.google.com/drive/folders/1jrn5CiZzkF1yWtmlD1Yaxr0b6nroM0Dy?usp=sharing

## Project Flow/Results 
   
The project flow defines how users interact with the system from start to finish—how data moves through each layer and how every component comes together to deliver meaningful sentiment insights. 
This sentiment analysis system was designed with usability and purpose in mind. Built on a modern stack and supported by deep learning models, it enables users to analyze the emotional tone behind YouTube comments with just a few simple interactions. 
 ![image](https://github.com/user-attachments/assets/88f628c4-f8e9-49c5-a536-3894e9a1b0a5)

  
Figure 1 - Landing Page 
- As the primary point of contact, the landing page serves not just as an entry screen, but as a welcoming space that sets the tone for the platform. It reflects a clean and visually consistent design, heavily inspired by YouTube’s branding to immediately resonate with the user. The prominent tagline- “Stream the Sentiment: Deep Learning Analysis”—communicates the platform’s intent while appealing to both technical and non-technical audiences. 
The layout is intentionally minimalistic, avoiding clutter while directing attention to key functions such as Signup, Login, and navigation. A dynamic sidebar on the left offers a bold yet functional menu that includes links to Home, Admin, Users, and Signup pages. The active use of gradients, color contrast, and iconography ensures that the interface remains both modern and user-friendly. 
 
  ![image](https://github.com/user-attachments/assets/3f9080a0-8c29-40ab-b557-6551f590897d)

Figure 2- User Video Id Input 
- Upon logging into the system, users are greeted with a clean and intuitive dashboard where they can input a YouTube video ID to initiate the sentiment analysis process. This streamlined interface is designed to be simple yet functional, guiding users to the core feature without distraction. When a video ID is submitted, the system fetches associated comments via the YouTube API and passes them through a robust preprocessing pipeline that includes cleaning, tokenization, and normalization. These prepared comments are then analyzed using deep learning models—primarily Bi-LSTM—to classify sentiments as positive, negative, or neutral. Results are returned in real-time, displayed in both textual and graphical formats to help users quickly interpret public sentiment surrounding the video. This seamless process transforms raw viewer feedback into meaningful insights in just a few clicks. 
  ![image](https://github.com/user-attachments/assets/20ebe729-8309-402e-9b00-40a398e497db)

Figure 3 - Checking Comments 
- This image showcases the "Crawl Comments" feature of the system, where users can input a YouTube video ID to fetch related comments in real-time using the YouTube Data API. Once the video ID is submitted, the system retrieves and displays the corresponding video title and thumbnail for context. Below this, the extracted comments begin to appear dynamically, ready for subsequent sentiment analysis. This functionality allows users to interactively engage with fresh, user-generated content and is a key step in analyzing audience reactions through deep learning models. 
  ![image](https://github.com/user-attachments/assets/a2ea12dc-4ef9-45f2-95a2-bb9aef22b52d)

Figure 4- Sentiment Analysis Results 
- Presents the sentiment analysis output of YouTube comments using the Bi-LSTM model, offering a rich and interactive interface for users to interpret results. At the center of the screen is a pie chart, clearly visualizing the sentiment distribution—positive, negative, and neutral—based on the analyzed comments. Accompanying this chart is relevant video metadata, such as the video title, thumbnail, and a direct link to the content. Below, individual comments are displayed alongside their predicted sentiment category, providing users with both a highlevel summary and detailed insights. This layered presentation makes it easy for users to understand public perception and viewer engagement at a glance. 
  ![image](https://github.com/user-attachments/assets/67f3310f-26cb-4935-acc3-b57ce979194e)

Figure 5 - Google API for Youtube 
- Illustrates the creation of an API key within the Google Cloud Console, a vital step in enabling secure and authenticated access to the YouTube Data API. This key allows the sentiment analysis system to retrieve comments tied to specific YouTube video IDs in real-time. By integrating this API key within the backend logic of the application, the system can dynamically fetch video metadata and user-generated comments, forming the foundation for subsequent semantic and sentiment analysis. The API key acts as a bridge between the user interface and the vast data resources of YouTube, ensuring the platform functions seamlessly and securely. 
  ![image](https://github.com/user-attachments/assets/8bc6a001-c11f-4f2f-b5f7-d61fb6e99bfa)

Figure 6 - Generating Comments from API 
- Showcases the terminal output generated during the execution of a Python script designed to retrieve YouTube comments using the YouTube Data API. This console log provides a clear view of the fetched video metadata, thumbnail URL, and the corresponding list of user comments tied to the input video ID. It demonstrates how the system successfully connects with the API, pulls structured comment data, and outputs it for processing. This terminal view is instrumental for developers during the implementation phase, helping verify that the API is functioning correctly, and the data is being parsed and prepared for further sentiment analysis. 
  ![image](https://github.com/user-attachments/assets/4bf78904-2c27-433c-ae57-b98ba766fea2)

Figure 7 - Database Management(SQLyog) 
- Illustrates the SQLyog Community Edition interface actively connected to the youtubecomments database used in the sentiment analysis system. This environment highlights the structured storage and management of user information, video metadata, and sentiment results. The left panel displays an organized schema with tables such as comments, users, and videos, allowing for efficient navigation. In the central workspace, we see SQL queries being composed and executed to interact with the data—like retrieving user signup information. Below, the live data view shows entries from the videos table, including video titles, associated user emails, and metadata like video IDs and thumbnail links. This setup not only simplifies data access and manipulation but also plays a crucial role in ensuring transparency, integrity, and traceability throughout the analysis pipeline. 
  ![image](https://github.com/user-attachments/assets/ef04af48-0a0a-492d-9bf5-783df8354334)

Figure 8 - Admin Homepage 
- Showcases the Admin Home Page of the sentiment analysis system, which functions as the operational backbone for managing datasets, triggering deep learning model training, and overseeing classification outputs. Upon successful login, administrators are welcomed with a streamlined dashboard that features a left-hand navigation panel linking to core functionalities such as Dataset Analysis, Model Classification, and Classification Results. The layout is intentionally minimal yet functional, enabling admins to quickly upload new datasets, initiate training processes for RNN, LSTM, GRU, and Bi-LSTM models, and monitor the results in real-time. This centralized interface ensures that administrative tasks can be carried out efficiently and securely, providing oversight and control without requiring deep technical expertise. 
 ![image](https://github.com/user-attachments/assets/3222b820-9d5d-4469-838d-3e1c32fd70f4)
 
Figure 9 - Dataset analysis 
- The Data Analysis section of the admin interface offers a visual and interactive representation of sentiment distribution across the dataset, empowering administrators with insights into the emotional landscape of collected YouTube comments. As depicted in the figure, the sentiment distribution is broken down into three distinct categories—positive, neutral, and negative—represented via a pie chart. This visual aid is not only userfriendly but also informative, enabling quick assessments of dataset balance and diversity. The admin page allows the uploading of new datasets, viewing of existing data, and real-time feedback on sentiment trends, all of which support informed decision-making regarding model training and performance optimization. 
  ![image](https://github.com/user-attachments/assets/4318a142-7ff3-46bd-9064-b807158debe8)

Figure 10 - Classification - Algorithms 
- The Classification interface on the admin portal serves as a centralized control panel for initiating and managing model training operations. Administrators can choose from a suite of advanced deep learning algorithms—RNN, LSTM, GRU, and Bi-LSTM—each optimized for sequential data analysis. With a simple click, the training process begins on the uploaded dataset, allowing the backend to execute model-specific training scripts. This modular structure not only offers flexibility in model experimentation but also supports comparative evaluation, making it easier for administrators to benchmark algorithm performance based on accuracy, precision, recall, and F1-score. 
  ![image](https://github.com/user-attachments/assets/b5894ad3-ebc5-4684-a517-f7bc3afad2a6)

Figure 11 - Classification Results 
- The Classification Results page provides a comprehensive summary of the model evaluation phase, displaying key performance metrics—accuracy, precision, recall, and F1-score—for each trained algorithm including RNN, LSTM, GRU, and Bi-LSTM. These metrics are presented both in tabular form and through visual aids like bar graphs, enabling a clear and intuitive comparison of model performance. This view helps administrators quickly identify the most effective model, with Bi-LSTM clearly outperforming others across all metrics. Such insights support data-driven decisions when selecting which model to deploy for real-time sentiment analysis in production.


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


