import string, nltk
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report, roc_auc_score, plot_roc_curve
import pickle, os, glob

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
    temp = [w for w in temp if w not in nltk.corpus.stopwords.words('english')]
    for i in range(len(temp)):
        temp[i] = stemmer.stem(temp[i])
    return ' '.join(temp)

def labelEncoder(dfcol):
    le = LabelEncoder().fit(dfcol)
    return le.transform(dfcol)

def createSplit(df, printsize=False):
    total_size = len(df)
    random_permutation = np.random.permutation(total_size)
    train_size = int(total_size*0.7)
    test_size = int((total_size - train_size)/2)
    val_size = test_size

    X_train = df['review'][random_permutation[:train_size]]
    y_train = np.array(df['sentiment'][random_permutation[:train_size]])

    X_test = df['review'][random_permutation[train_size:train_size+test_size]]
    y_test = np.array(df['sentiment'][random_permutation[train_size:train_size+test_size]])

    X_val = df['review'][random_permutation[-test_size:]]
    y_val = np.array(df['sentiment'][random_permutation[-test_size:]])

    if printsize==True:
        print("Training set : Validation set : Test set = "+str(len(X_train))+" : "+str(len(X_val))+" : "+str(len(X_test)))

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def trainClassifier(clf, ngram_lb, ngram_ub, train_X, train_y, val_X, val_y):
    tfidf = TfidfVectorizer(ngram_range=(ngram_lb, ngram_ub), max_features=None, sublinear_tf=True) #stop_words=nltk.corpus.stopwords.words('english')
    pipe = Pipeline([('tfidf', tfidf), ('classifier', clf)])
    pipe.fit(train_X.astype(str), train_y)
    acc = accuracy_score(val_y, pipe.predict(val_X))
    f1acc = np.mean(cross_val_score(pipe, val_X, val_y, scoring=make_scorer(f1_score), cv=10))
    return {
        'pipeline' : pipe,
        'acc' : acc,
        'f1_cv' : f1acc
    }

def saveModel(clf, model, path):
    name = f"{clf}-{model['f1_cv']*100:.3f}.sav"
    pickle.dump(model['pipeline'], open(os.path.join(path, name), 'wb'))


def printReport(X_test, y_test, model, roc=False):
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"Prediction for {model.split('/')[-1].split('-')[0]}")
    model = pickle.load(open(model, 'rb'))
    y_pred = model.predict(X_test)
    print(f"Accuracy Score : {accuracy_score(y_test, y_pred):.3f}\nF1 Score : {f1_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    if roc==True:
        plot_roc_curve(model, X_test, y_test)

def getBestModel(path, name, overwrite):
    models = glob.glob(os.path.join(path, '*.sav'))
    models = [m for m in models if m.split('/')[-1].startswith(name)]
    models = sorted(models, key=lambda x: float(x.split('-')[-1][:-4]), reverse=True)

    if overwrite==True:
        if len(models)>1:
            for f in models[1:]:
                os.remove(f)
    
    return pickle.load(open(models[0], 'rb'))