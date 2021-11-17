from utils import *
from models import *
import pandas as pd
import glob, nltk
nltk.download('stopwords')

df = pd.read_csv('IMDB Dataset.csv')
print("Preprocessing") #
df['review'] = df['review'].apply(preprocess)
df['sentiment'] = labelEncoder(df['sentiment'])

print("Splitting") #
X_train, y_train, X_val, y_val, X_test, y_test = createSplit(df, printsize=True)

path = './Models/'
models = glob.glob(os.path.join(path, '*.sav'))

# If no model is there. It can also be run without if-block because it can replace the existing models
# overwrite=True
if len(models)==0:
    getLinearSVM(X_train, y_train, X_val, y_val, path, progress=True),
    getLogisticRegressor(X_train, y_train, X_val, y_val, path, progress=True)
    models = glob.glob(os.path.join(path, '*.sav'))

print("\nModel-wise Prediction")
for model in models:
    printReport(model=model, X_test=X_test, y_test=y_test, roc=True)

print("\nCommittee Prediction")
y_pred = getCommitteePrediction(path, X_test)
printReport(y_test=y_test, y_pred=y_pred)