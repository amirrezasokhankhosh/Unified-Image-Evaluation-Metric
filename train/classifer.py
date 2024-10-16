import os
import numpy as np
import tensorflow as tf


class Classifier:
    def __init__(self, dataset, latent_dim=100):
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.get_data(dataset)
        self.get_model()

    def get_data(self, dataset):
        if dataset == "mnist":
            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        elif dataset == "fashion":
            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        self.X_train = np.expand_dims(self.X_train, -1).astype("float32") / 255
        self.X_test = np.expand_dims(self.X_test, -1).astype("float32") / 255

    def get_model(self):
        classifier_input = tf.keras.layers.Input((28, 28, 1), name="classifier_input")
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(classifier_input)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        latent_features = tf.keras.layers.Dense(self.latent_dim)(x)
        x = tf.keras.activations.relu(latent_features)
        classifier_output = tf.keras.layers.Dense(10, activation='softmax', name="classifier_output")(x)
        self.model = tf.keras.models.Model(classifier_input, [classifier_output, latent_features])
    
    def train(self):
        self.model.compile(optimizer='adam', 
                   loss=['sparse_categorical_crossentropy', None],
                   loss_weights=[1.0, 0.0],
                   metrics=['accuracy'])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)
        self.model.fit(self.X_train, self.y_train, 
                        batch_size=32, epochs=100, 
                        validation_data=(self.X_test, self.y_test), 
                        callbacks=[early_stopping])
        self.model.save(f"../models/{self.dataset}/classifier.keras")

    def load_model(self):
        cwd = os.path.dirname(__file__)
        model_path = os.path.abspath(os.path.join(cwd, f"../models/{self.dataset}/classifier.h5"))
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', 
                   loss=['sparse_categorical_crossentropy', None],
                   loss_weights=[1.0, 0.0],
                   metrics=['accuracy'])
