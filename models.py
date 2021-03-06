from utils import trainClassifier, saveModel, getBestModel
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

def getLinearSVM(X_train, y_train, X_val, y_val, path, progress=False, overwrite=True):
    
    #C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    C = [0.1, 1, 10]

    if os.path.exists(path)==False:
        os.mkdir(path)

    best_model = None
    best_f1 = 0
    ngram_lb = 1
    ngram_ub = 2

    if progress==True:
        print("\n----- LINEAR SVM -----\n")
        print("C\tAccuracy\tF1-Score\n===================================")
        for c in C :
            clf = LinearSVC(C=c, multi_class='ovr', class_weight='balanced', max_iter=100000)
            for ngram_ub in range(1,5):
                temp = trainClassifier(clf, ngram_lb, ngram_ub, X_train, y_train, X_val, y_val)
                print(f"{c}\t{temp['acc']:.3f}\t\t{temp['f1_cv']:.3f}")
                if temp['f1_cv'] <= best_f1:
                    continue
                best_f1 = temp['f1_cv']
                best_model = temp
    else:
        for c in C :
            clf = LinearSVC(C=c, multi_class='ovr', class_weight='balanced', max_iter=100000)
            for ngram_ub in range(1,5):
                temp = trainClassifier(clf, ngram_lb, ngram_ub, X_train, y_train, X_val, y_val)
                if temp['f1_cv'] <= best_f1:
                    continue
                best_f1 = temp['f1_cv']
                best_model = temp
    saveModel('linsvm', best_model, path)
    return getBestModel(path, 'linsvm', overwrite)

def getLogisticRegressor(X_train, y_train, X_val, y_val, path, progress=False, overwrite=True):

    #C = [0.0001, 0.001, 0.001, 0.01, 0.1, 1, 10, 100]
    C = [0.1, 1, 10]

    if os.path.exists(path)==False:
        os.mkdir(path)

    best_model = None
    best_f1 = 0
    ngram_lb = 1
    ngram_ub = 2

    if progress==True:
        print("\n----- LOGISTIC REGRESSION -----\n")
        print("C\tAccuracy\tF1-Score\n===================================")
        for c in C :
            clf = LogisticRegression(C=c, multi_class='auto', class_weight='balanced', max_iter=100000)
            for ngram_ub in range(1,5):
                temp = trainClassifier(clf, ngram_lb, ngram_ub, X_train, y_train, X_val, y_val)
                print(f"{c}\t{temp['acc']:.3f}\t\t{temp['f1_cv']:.3f}")
                if temp['f1_cv'] <= best_f1:
                    continue
                best_f1 = temp['f1_cv']
                best_model = temp
    else:
        for c in C :
            clf = LogisticRegression(C=c, multi_class='auto', class_weight='balanced', max_iter=100000)
            for ngram_ub in range(1,5):
                temp = trainClassifier(clf, ngram_lb, ngram_ub, X_train, y_train, X_val, y_val)
                if temp['f1_cv'] <= best_f1:
                    continue
                best_f1 = temp['f1_cv']
                best_model = temp
    saveModel('logreg', best_model, path)
    return getBestModel(path, 'logreg', overwrite)

def getMNBClassifier(X_train, y_train, X_val, y_val, path, progress=False, overwrite=True):
    
    #C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    C = [0.1, 1, 10]

    if os.path.exists(path)==False:
        os.mkdir(path)

    best_model = None
    best_f1 = 0
    ngram_lb = 1
    ngram_ub = 4

    if progress==True:
        print("\n----- MULTINOMIAL NAIVE BAYES -----\n")
        print("C\tAccuracy\tF1-Score\n===================================")
        for c in C :
            clf = MultinomialNB(alpha=c)
            for ngram_ub in range(1,5):
                temp = trainClassifier(clf, ngram_lb, ngram_ub, X_train, y_train, X_val, y_val)
                print(f"{c}\t{temp['acc']:.3f}\t\t{temp['f1_cv']:.3f}")
                if temp['f1_cv'] <= best_f1:
                    continue
                best_f1 = temp['f1_cv']
                best_model = temp
    else:
        for c in C :
            clf = MultinomialNB(alpha=c)
            for ngram_ub in range(1,5) :
                temp = trainClassifier(clf, ngram_lb, ngram_ub, X_train, y_train, X_val, y_val)
                if temp['f1_cv'] <= best_f1:
                    continue
                best_f1 = temp['f1_cv']
                best_model = temp
    saveModel('multinb', best_model, path)
    return getBestModel(path, 'multinb', overwrite)

def getKNNClassifier(X_train, y_train, X_val, y_val, path, progress=False, overwrite=True):
    
    n_neighbors = [i for i in range(3, 10, 2)]

    if os.path.exists(path)==False:
        os.mkdir(path)

    best_model = None
    best_f1 = 0
    ngram_lb = 1
    ngram_ub = 2

    if progress==True:
        print("\n----- KNN -----\n")
        print("NN\tAccuracy\tF1-Score\n===================================")
        for nn in n_neighbors :
            clf = KNeighborsClassifier(n_neighbors=nn)
            temp = trainClassifier(clf, ngram_lb, ngram_ub, X_train, y_train, X_val, y_val)
            print(f"{nn}\t{temp['acc']:.3f}\t\t{temp['f1_cv']:.3f}")
            if temp['f1_cv'] <= best_f1:
                continue
            best_f1 = temp['f1_cv']
            best_model = temp
    else:
        for nn in n_neighbors :
            clf = KNeighborsClassifier(n_neighbors=nn)
            temp = trainClassifier(clf, ngram_lb, ngram_ub, X_train, y_train, X_val, y_val)
            if temp['f1_cv'] <= best_f1:
                continue
            best_f1 = temp['f1_cv']
            best_model = temp
    saveModel('knn', best_model, path)
    return getBestModel(path, 'knn', overwrite)