# Disaster Response Pipeline Project

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Data](#data)
4. [File Descriptions](#files)
5. [Results](#results)

## Installation <a name="installation"></a>
You will need the standard data science libraries like pandas, numpy, NLTK and scikit learn to run the  project. For the visualisations, installing plotly will be necessary. 

## Project Motivation <a name="motivation"></a>

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
