import numpy as np
from math import sqrt


class KnnClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self
    
    def predict(self, X_test) -> np.ndarray:
        return np.array([self.get_type(x_test) for x_test in np.array(X_test)])
    
    def get_type(self, x_test: np.ndarray):
        dists = [(self.dist(x_test, self.X_train[i]), self.X_train[i]) for i in range(len(self.X_train))]
        neighbors = [sorted(dists, key=lambda x: x[0])[i][1] for i in range(self.n_neighbors)]
        
        types = {}
        for neighbor in neighbors:
            if (t := self.y_train[np.where(np.all(self.X_train == neighbor, axis=1))[0][0]]) in types:
                types[t] += 1
            else:
                types[t] = 1
        
        return max(types.items(), key=lambda x: x[1])[0]
    
    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return sqrt(np.sum((a - b)**2))
