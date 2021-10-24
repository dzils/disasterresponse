# Disaster Response Pipeline Project

## Motivation

## Results

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
    

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/
