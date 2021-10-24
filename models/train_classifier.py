import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    Loads a sqlite database file.

    :param database_filepath: path to the database file
    :return:
        X: Contains the messages
        Y: Contains the categories the corresponding X values
        category_names: List of all category names
    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)
    
    X = df['message'].values
    Y = df.drop(columns=['id','message','original','genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    Returns tokens based on given text. 
    The text is transformed to lower case and punctuation is removed.
    Also, the tokens are lemmatized and stop words are removed
    
    Parameters:
    text (string): the text to be tokenized
    
    Returns:
    The tokenized text
    '''

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    Returns a GridSearchCV containing a Pipeline which uses a CountVectorizer, a TfidfTransformer and a MultiOutputClassifier
    :return: The GridSearchCV
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])
    
    parameters = {'clf__estimator__learning_rate': [0.25, 0.33, 0.66, 1],
                 'clf__estimator__n_estimators': [50, 75, 100]}
    
    return GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints a classification report for the given model and test values.

    :param model: model to be tested
    :param X_test: X test values
    :param Y_test: Y test values
    :param category_names: category labels for Y
    '''

    print(classification_report(Y_test, y_pred = model.best_estimator_.predict(X_test), target_names=category_names))


def save_model(model, model_filepath):
    '''
    Exports a model to a pickle file.

    :param model: model to be exported
    :param model_filepath: the filepath where the model is exported to
    '''

    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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
