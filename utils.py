import string, nltk
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from matplotlib import pyplot as plt
import pickle, os

def preprocess(text):
    soup = BeautifulSoup(text, "html.parser")
    for data in soup(['script', 'style']):
        data.decompose()
    text = ' '.join(soup.stripped_strings)
    text = text.lower()
    temp = ""
    for i in text:
        if i in string.punctuation:
            continue
        else:
            temp+=i
    temp = temp.split()
    for i in range(len(temp)):
        temp[i] = nltk.stem.PorterStemmer().stem(temp[i])
    return ' '.join(temp)

def labelEncoder(dfcol):
    le = LabelEncoder().fit(dfcol)
    return le.transform(dfcol)

def createSplit(df):
    total_size = len(df)
    random_permutation = np.random.permutation(total_size)
    train_size = int(total_size*0.7)
    test_size = int((total_size - train_size)/2)
    val_size = test_size

    X_train = df['review'][random_permutation[:train_size]]
    y_train = df['sentiment'][random_permutation[:train_size]]

    X_test = df['review'][random_permutation[train_size:train_size+test_size]]
    y_test = df['sentiment'][random_permutation[train_size:train_size+test_size]]

    X_val = df['review'][random_permutation[-test_size:]]
    y_val = df['sentiment'][random_permutation[-test_size:]]

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def trainClassifier(clf, ngram, train_X, train_y, val_X, val_y):
    if val_y is None or train_y is None:
        print("None")
    count = CountVectorizer(ngram_range=(1, ngram), stop_words=nltk.corpus.stopwords.words('english'))
    tfidf = TfidfTransformer()
    pipe = Pipeline([('count', count), ('tfidf', tfidf), ('classifier', clf)])
    start = datetime.now()
    pipe.fit(train_X.astype(str), train_y)
    end = datetime.now()
    acc = accuracy_score(val_y, pipe.predict(val_X))
    f1acc = np.mean(cross_val_score(pipe, val_X, val_y, scoring=make_scorer(f1_score), cv=10, n_jobs=-1))
    return {
        'pipeline' : pipe,
        'c' : c,
        'ngram' : ngram,
        'acc' : acc,
        'f1_cv' : f1acc
    }

def saveModel(clf, model, path):
    name = f"{clf}-{model['f1_cv']*100:.3f}.sav"
    pickle.dump(model['pipeline'], open(os.path.join(path, name), 'wb'))