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




def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("etl", engine)
    X = df.message
    Y = df[['related', 'offer','aid_related']]
    cat_names = df.columns[4:]
    print(list(cat_names))
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
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    


def save_model(model, model_filepath):
    pass


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