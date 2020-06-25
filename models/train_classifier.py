import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle




def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("etl", engine)
    X = df.message
    Y = df[['related', 'offer','aid_related']]
    cat_names = df.columns[4:]
    Y = df[list(cat_names)]
    print(list(cat_names))
    print(X.shape)
    return X, Y, list(cat_names)


def tokenize(text):
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]', ' ', text)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        #print(clean_tok)
        clean_tokens.append(clean_tok)
    clean_tokens = [w for w in tokens if w not in stopwords.words('english')]

    return clean_tokens



def build_model():
    pipeline = Pipeline([
    ('vect' , CountVectorizer(tokenizer=tokenize, ngram_range =(1, 1))),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    
    ])
    parameters =  {
              'clf__estimator__n_estimators': [50, 100]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2,verbose=10, n_jobs=-1)
    return cv

def make_report(y_true, y_pred):
    """
    Generate a performance report for a multiclass prediction
    INPUT:
        y_true - true labels
        y_pred - predicted labels

    """
    for i, col in enumerate(list(y_true.columns)):
        print(col)
        report = classification_report(y_true[col], y_pred[:,i], output_dict=False, zero_division=0)
        print(report)

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model and generates performance report
    Parameters:
        model - the ML model
        X_test - test data
        Y_test - labels from test data
        category_names - list of category names
    """
    y_pred = model.predict(X_test)
    make_report(Y_test, y_pred)
    



def save_model(model, model_filepath):
    """
    saves mode as pickle file

    Parameters:
        model - the ML model to export
        model_filepath - file path of the pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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