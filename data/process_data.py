import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads and merges the data from messages and categories csv files.

    :param messages_filepath: path to messages csv file
    :param categories_filepath: path to categories csv file
    :return: a dataframe with messages and categories merged together
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, how='inner', on='id')


def clean_data(df):
    '''
    Cleans a dataframe and removes duplicates.

    :param df: dataframe to be cleaned
    :return: a cleaned dataframe
    '''

    categories = df.categories.str.split(pat=';', expand=True)
    # rename category column names
    row = categories.iloc[0]
    category_colnames = row.apply(func=lambda string : string[0:len(string)-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(func=lambda string : string[len(string)-1:len(string)])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop original categories column and add the new ones
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    # clean up unexpected values
    for column in categories:
        df = df.loc[df[column] <= 1 ]
        
    return df

def save_data(df, database_filename):
    '''
    Stores a dataframe into a sqlite database file.

    :param df: the datadrame to be stored
    :param database_filename: filename of the database file
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


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
