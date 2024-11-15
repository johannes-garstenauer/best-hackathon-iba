from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split


class DataContainer:
    def __init__(self, path_to_data, path_to_labels):
        self.X = np.load(path_to_data)
        self.y = np.load(path_to_labels)
        self.transformer = Normalizer().fit(self.X)
        self.scaler = StandardScaler().fit(self.X)

    def normalize_data(self):
        self.X = self.transformer.transform(self.X)

    def scale(self):
        self.X = self.scaler.transform(self.X)
    
    def get_data(self):
        return self.X
    
    def get_labels(self):
        return self.y
    
    def print_scape(self):
        print(self.X.shape)
        

dataContainer = DataContainer("./data/test_data.npy", "./data/test_labels.npy")
dataContainer.normalize_data()
dataContainer.scale()

s = 100000

X = dataContainer.get_data()[:s]
y = dataContainer.get_labels()[:s]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

model = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.15)
model.fit(X_train)
y_predict = model.predict(X_test)
y_predict = (y_predict == -1).astype(int)

print(np.unique(y_predict, return_counts=True))

accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)

print(
f"""
Accuracy :  {accuracy}
f1       :  {f1}
precision:  {precision}
recall   :  {recall}
""")