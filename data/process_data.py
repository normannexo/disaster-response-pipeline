import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """
    loads messages and categories data from given file paths

    Parameters:
        messages_filepath (str): the path to the messages csv file
        categories_filepath (str): the path to the categories csv file
    Returns:
        data frame: a pandas data frame with the merged messages and category data
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath )
    #merge the data sets:
    df = categories.merge(messages, on ='id')
    return df


def clean_data(df):
    """
    cleans the data frame loaded from the csv raw data

    Parameters:
        df: the pandas data frame loaded with load_data()
    Returns:
        data frame: a pandas data frame with cleaned data
    """
    # split values in the categories columns on ';' into different columns
    categories = df['categories'].str.split(";", expand=True)
    # select first row of the categories data frame and extract list
    # of new column names
    row = categories.iloc[0].apply(lambda x: x[:-2])
    category_colnames = row.values
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    return df



def save_data(df, database_filename):
    """
    saves data to sqlite data base
    Parameters:
        df - pandas data frame to export
        database_filename - the filename of the sqlite database
    """
    try:
        engine = create_engine('sqlite:///' + database_filename)
        df.to_sql('etl', engine, index = False)
    except Exception as e:
        print(e)


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