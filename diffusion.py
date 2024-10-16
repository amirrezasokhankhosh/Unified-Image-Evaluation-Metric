from pytorch_diffusion import Diffusion
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.neighbors import NearestNeighbors
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class ResNet50CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50CIFAR10, self).__init__()
        # Load pre-trained ResNet-50 model
        self.model = resnet50(pretrained=True)
        
        # Modify the fully connected layer to match the number of CIFAR-10 classes (10)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x, return_feature=False):
        # Forward pass through the ResNet layers up to the global average pooling layer
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # Global average pooling
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        # If return_feature is True, return the output before the final classifier layer
        if return_feature:
            return x

        # Otherwise, pass through the classifier and return the final prediction
        x = self.model.fc(x)
        return x

class GeneralizationRate:
    def __init__(self, train_features, test_features, delta=0.0001):
        self.k = 1
        self.delta = delta

        self.train_features = train_features
        self.test_features = test_features
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

class Precision:
    def __init__(self, train_features, test_features, delta=0.0001):
        self.k = 1
        self.classifier = classifier
        self.delta = delta

        self.train_features = train_features
        self.test_features = test_features

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
            if distances_samples[i] < distances_train[train_indice, self.k-1]:
                count += 1
        return count / len(gen_features)

    def calculate_k(self):
        """
        Using test data, I calculate the right number for k according to our custom delta value.
        To do so, I increase the value of k from 1 until the generalization rate becomes less than delta.
        """
        while True:
            os.system("clear")
            print("Calculating K for precision:")
            print(f"Current K : {self.k}")
            rate = self.get(self.test_features)
            if 1 - rate < self.delta:
                break
            self.k += 1

class Recall:
    def __init__(self, train_features, test_features, delta=0.0001):
        self.k = 1
        self.delta = delta

        self.train_features = train_features
        self.test_features = test_features

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

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

# Download test data from open datasets.
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=len(training_data))
X_train, _ = next(iter(train_dataloader))
test_dataloader = DataLoader(test_data, batch_size=len(test_data))
X_test, _ = next(iter(test_dataloader))

# Example instantiation of the model
classifier = ResNet50CIFAR10(num_classes=10).to(device)
classifier.load_state_dict(torch.load("./CIFAR10-Classifier.pth", map_location=torch.device('cpu')))

print("Sampling...")
diffusion = Diffusion.from_pretrained("cifar10")
samples = diffusion.denoise(1000)

gen_features = classifier(samples, return_feature=True)
train_features = classifier(X_train, return_feature=True)
test_features = classifier(X_test, return_feature=True)

gen_rate = GeneralizationRate(classifier=classifier, X_train=X_train, X_test=X_test, delta=0.001).get(gen_features)
precision = Precision(classifier=classifier, X_train=X_train, X_test=X_test, delta=0.001).get(gen_features)
recall = Recall(classifier=classifier, X_train=X_train, X_test=X_test, delta=0.001).get(gen_features)
f1 = (3*recall*precision*gen_rate)/(recall+precision+gen_rate)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Generalization Rate: {gen_rate}")
print(f"F1: {f1}")