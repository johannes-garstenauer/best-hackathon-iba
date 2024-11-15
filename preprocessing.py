import numpy as np
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

##for feature in range(data.shape[1]):
##    column = data[:, feature]
##    z_scores = np.abs((column - column.mean()) / column.std())
##    outliers = (z_scores > 5)
##    outlier_counts = outliers.sum()
##    print(outlier_counts)

s = 10000

X = dataContainer.get_data()[:s]
y = dataContainer.get_labels()[:s]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

for eps in range(1,500):
    eps = eps/1000.0
    dbscan = DBSCAN(eps=eps, min_samples=5)

    y_predict = dbscan.fit_predict(X_train)
    y_predict = (y_predict == -1).astype(int)

    accuracy = accuracy_score(y_train, y_predict)
    f1 = f1_score(y_train, y_predict)
    precision = precision_score(y_train, y_predict)
    recall = recall_score(y_train, y_predict)

    print(eps, np.unique(y_predict).size)
    print(
f"""
Accuracy :  {accuracy}
f1       :  {f1}
precision:  {precision}
recall   :  {recall}
""")
    print()
