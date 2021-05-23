# Disaster Response Pipeline Project

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Data](#data)
4. [Project Components](#project_components)
5. [How to Run the Project](#run)

## Installation <a name="installation"></a>
You will need the standard data science libraries like pandas, numpy, NLTK and scikit learn to run the  project. For the visualisations, installing plotly will be necessary. 

## Project Motivation <a name="motivation"></a>
Following a disaster, there could be millions of communications (on social media or directly sent to the disaster response agencies) where only a few will be relevant to the disaster response organisations. To direct the message to the relevant organisation, we need to understand what the message is about: e.g., water, medical supplies, electricity, infrastructure. This is what this project tries to achieve. 

## Dataset <a name="data"></a>
The data sources are `messages.csv` and `categories.csv`, compiled and prepared by FigureEight for Machine Learning tasks. 

## Project Components <a name="project_components"></a>

1. ETL Pipeline
    - Loads `messages.csv` and `categories.csv` datasets
    - Merges the two datasets
    - Cleans the data
    - Stores the cleaned data in a SQLite database

2. Machine Learning Pipeline
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the tuned model as a pickle file

3. Flask Web App

## How to Run the Project <a name="run"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
