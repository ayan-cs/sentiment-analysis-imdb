from utils import *
from models import *
import pandas as pd
import glob, nltk
nltk.download('stopwords')

if os.path.exists('./IMDb_stemmed.csv')==False:
    df = pd.read_csv('IMDB Dataset.csv')
    print("Preprocessing dataset! This may take a while ...") #
    df['review'] = df['review'].apply(preprocess)
    df['sentiment'] = labelEncoder(df['sentiment'])
    df.to_csv('IMDB_stemmed.csv')
else:
    df = pd.read_csv('IMDB_stemmed.csv')

print("Splitting dataset")
X_train, y_train, X_val, y_val, X_test, y_test = createSplit(df, printsize=True)

path = './Models/'
if os.path.exists(path)==False:
    os.mkdir(path)
models = glob.glob(os.path.join(path, '*.sav'))
vects = glob.glob(os.path.join(path, '*.pk'))

# If no model is there. It can also be run without if-block because it can replace the existing models
# overwrite=True
if len(models)==0:
    getLinearSVM(X_train, y_train, X_val, y_val, path, progress=True, overwrite=True),
    getLogisticRegressor(X_train, y_train, X_val, y_val, path, progress=True, overwrite=True)
    getMNBClassifier(X_train, y_train, X_val, y_val, path, progress=True, overwrite=True)
    models = glob.glob(os.path.join(path, '*.sav'))
    vects = glob.glob(os.path.join(path, '*.pk'))

print(models,'\n',vects)

print("\nModel-wise Prediction Report\n")
for i in range(len(models)):
    model = {'clf' : models[i], 'vect' : vects[i]}
    printReport(model=model, X_test=X_test, y_test=y_test, roc=True)