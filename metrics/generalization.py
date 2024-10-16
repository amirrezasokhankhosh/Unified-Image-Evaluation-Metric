import numpy as np
from sklearn.neighbors import NearestNeighbors
import os


class GeneralizationRate:
    def __init__(self, classifier, X_train, X_test, delta=0.0001):
        self.k = 1
        self.classifier = classifier
        self.delta = delta

        self.train_features = self.classifier(X_train, return_feature=True)
        self.test_features = self.classifier(X_test, return_feature=True)
        self.min_space = np.min(self.train_features, axis=0, keepdims=True)
        self.max_space = np.max(self.train_features, axis=0, keepdims=True)

        self.calculate_k()

    def get(self, gen_features):
        """
        Calculates Generalization
        """
        nn = NearestNeighbors(n_neighbors=self.k+1).fit(self.train_features)
        distances_train, _ = nn.kneighbors(self.train_features)
        distances_samples, indices_samples = nn.kneighbors(gen_features, 1)

        count = 0
        for i in range(len(gen_features)):
            train_indice = indices_samples[i]
            if distances_samples[i] > distances_train[train_indice, self.k-1] and \
                    np.all(gen_features[i] > self.min_space) and \
                    np.all(gen_features[i] < self.max_space):
                count += 1
        return count / len(gen_features)

    def calculate_k(self):
        """
        Using test data, I calculate the right number for k according to our custom delta value.
        To do so, I increase the value of k from 1 until the generalization rate becomes less than delta.
        """
        while True:
            os.system("clear")
            print("Calculating K for Generalization Rate:")
            print(f"Current K : {self.k}")
            rate = self.get(self.test_features)
            if rate < self.delta:
                break
            self.k += 1
