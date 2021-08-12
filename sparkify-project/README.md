### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Data](#data)
4. [File Descriptions](#files)
5. [Results](#results)

## Libraries <a name="installation"></a>
This project uses uses Spark MLLib library, pandas and also matplotlib for visualusations. 

## Project Motivation <a name="motivation"></a>
This project is about identifying customers of a music streaming service Sparkify who are likely to churn. 
Predicting churn is a fairly common task data scientists have to solve, and Spark is a highly efficient way of modelling 
the problem when dealing with large datasets.

## Data<a name="data"></a>
The classification here is based on a small dataset provided by Udacity (`mini_sparkify_event_data.json`) which contains logs of 255 users. 

## File Descriptions<a name="files"></a>
The repository contains the following files:

* `Sparkify.ipybn` - Jupyter notebook of the classification task using a small data set. It includes data loading, cleaning, data exploration, feature engineering, 
data modelling, evaluation, hyperparameter optimisation and feature importances. 
* `Sparkify_emr.ipybn` - Jupyter notebook of the classification task using the full (12GB) data set (`s3n://udacity-dsnd/sparkify/sparkify_event_data.json`). It includes all as above but data exploration.

## Results<a name="results"></a>
The main findings have been summarised in the blog post [here](https://katya-saintamand.medium.com/).
