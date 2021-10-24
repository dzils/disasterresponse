# Disaster Response Pipeline Project

## Motivation
The goal of this project is to classify disaster related messages from various sources in order to reduce the workload of a downstream disaster relief agency. 
To achieve this, a Language Processing Pipeline (NLP) is created. The NLP creates a machine learning model that is used in an Web Application which then can be used to categorize incoming messages.

## Results
The Web Application works as intended, although the model does not apply all categories. 
Example (taken from the csv files described below):

***There's nothing to eat and water, we starving and thirsty.***
  
**Should be:**  
related, request, aid_related, medical_help, medical_products, water, food, other_aid, infrastructure_related, transport, buildings, other_infrastructure, weather_related, floods, direct_report  

**Model result:**  
related, request, aid_related, water, food, direct_report  

While not all categories are recognized, the basic ones are set correctly and would indeed help to reduce the workload.

## This project was created with the following environment:
- python 3.8.10
- pandas 1.2.4
- scikit-learn 0.24.2
- sqlalchemy 1.4.22
- sqlite 3.35.4
- nltk 3.6.5
- plotly 5.1.0
- flask 1.1.2
- joblib 1.0.1

## Files & Data
 The data was provided by [Figure Eight](https://www.figure-eight.com/) in cooperation with [Udacity](https://www.udacity.com/).
 #### - ./data
 Contains the csv files for the messages and categories & the ETL pipeline
 #### - ./models
 Contains the NLP pipeline for creating the model used in the app.
 #### - ./app
 Contains the application and the HTML templates

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/
