import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import pickle
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\harsh\Downloads\Final_use.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=53)

# print(X_train)
# print(y_test)
print(np.shape(X_train))

classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
