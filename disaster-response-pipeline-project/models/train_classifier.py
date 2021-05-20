import sys

import sqlite3
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, average_precision_score    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    table_name='DisasterResponse'
    df = pd.read_sql(table_name, con=engine)

    X = df['message']

    categories = ['related', 'request', 'offer', 'aid_related', 
                  'medical_help', 'medical_products', 'search_and_rescue',
                  'security', 'military', 'child_alone', 'water', 'food', 
                  'shelter', 'clothing', 'money', 'missing_people', 
                  'refugees', 'death', 'other_aid', 'infrastructure_related', 
                  'transport', 'buildings', 'electricity', 'tools', 
                  'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                  'weather_related', 'floods', 'storm', 'fire', 'earthquake', 
                  'cold', 'other_weather', 'direct_report'
             ]
    
    y = df[categories]
    return X, y, categories

def tokenize(text):
    stopwords_en = stopwords.words('english')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords_en:
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    rfc_pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('rfc', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    rfc_parameters = {  'rfc__estimator__max_depth': [2, 3, 4],
                    'rfc__estimator__criterion' : ['gini', 'entropy'], 
                    'rfc__estimator__max_features' : ['auto', 'sqrt', 'log2'],
                    'rfc__estimator__class_weight' : ['balanced', 'balanced_subsample'],
              }

    rfc_cv = GridSearchCV(estimator=rfc_pipeline, 
                  param_grid=rfc_parameters)

        
   
    return rfc_cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(data=y_pred,columns=Y_test.columns)
    #score = f1_score(y_test,y_pred, average='micro')
    f1_scores = list()
    for i in range(Y_test.shape[1]):
        print("F1 score for each category")
        print("===========================================================")
        f1 = f1_score(Y_test.iloc[:,i],y_pred_df.iloc[:,i],average='micro')
        print(f1)
        f1_scores.append(f1)
    print("===========================================================")
    print("Total mean F1 for all categories")    
    print(np.mean(f1_scores)) 


def save_model(model, model_filepath):
    # save model
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()