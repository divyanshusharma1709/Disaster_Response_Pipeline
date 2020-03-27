import sys
from sqlalchemy import create_engine
import pandas as pd
import  re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def load_data(database_filepath):
    """Load table from SQL database and prepare Features and Labels"""
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response",engine)
    ls = df.columns[4:]
    Y = None
    for el in ls:
        Y = pd.concat([Y, df[el]], axis = 1)
    X = df['message']
    Y["related"].replace({2: 0}, inplace=True)
    return X, Y, Y.columns.values


def tokenize(text):
    """Remove Punctuation, Case Normalize the text and perform Lemmatization"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    return lemmed


def build_model():
    """Build ML Pipeline and use GridSearchCV"""
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', RandomForestClassifier())])
    parameters = {
    'clf__max_depth': [None, 10, 15, 12],
    'clf__n_estimators': [10, 15]
    }
    cv_g = GridSearchCV(pipeline, param_grid = parameters)
    return cv_g


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model on the test set"""
    pred = model.predict(X_test)
    print(classification_report(Y_test.values, pred, target_names = Y.columns.values))


def save_model(model, model_filepath):
    """Save the model to use it on the web app"""
    joblib.dump(model, model_filepath, compress = True)


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
