import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Reads categories and messages csv files, 
    merges them and loads to pandas dataframes'''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    
    return df


def clean_data(df):
    '''Reads in a dataframe and returns a cleaned up dataframe:
        categories are binary fields
        duplicates get dropped'''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    categories.head()
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split("-").str[0]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column],errors='coerce')

    # Replace categories column in df with new category columns.
    df=df.drop(columns='categories')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df=df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''Saves dataframe to a SQLite database'''

    engine = create_engine('sqlite:///./'+ str (database_filename))
    df.to_sql('disaster_response_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()