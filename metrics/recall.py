import numpy as np
from sklearn.neighbors import NearestNeighbors
import os


class Recall:
    def __init__(self, classifier, X_train, X_test, delta=0.0001):
        self.k = 1
        self.classifier = classifier
        self.delta = delta

        self.train_features = self.classifier(X_train, return_feature=True)
        self.test_features = self.classifier(X_test, return_feature=True)

        self.calculate_k()

    def get(self, gen_features):
        """
        Calculates Generalization
        """
        nn = NearestNeighbors(n_neighbors=self.k+1).fit(gen_features)
        distances_samples, _ = nn.kneighbors(gen_features)
        distances_train, indices_train = nn.kneighbors(self.train_features, 1)

        count = 0
        for i in range(len(self.train_features)):
            sample_indice = indices_train[i]
            if distances_train[i] < distances_samples[sample_indice, self.k-1]:
                count += 1 
        return count / len(self.train_features)

    def calculate_k(self):
        """
        Using test data, I calculate the right number for k according to our custom delta value.
        To do so, I increase the value of k from 1 until the generalization rate becomes less than delta.
        """
        while True:
            os.system("clear")
            print("Calculating K for recall:")
            print(f"Current K : {self.k}")
            rate = self.get(self.test_features)
            if 1 - rate < self.delta:
                break
            self.k += 1
