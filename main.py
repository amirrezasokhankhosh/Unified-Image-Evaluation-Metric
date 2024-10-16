import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from metrics.precision import Precision
from metrics.recall import Recall
from metrics.generalization import GeneralizationRate

class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
    def generate_images(self, batch_size=100):
        latent_size = 100

        fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
        generated_images = self.forward(fixed_noise)

        generated_images = generated_images.cpu().detach().numpy()
        return generated_images.reshape(generated_images.shape[0], 28, 28, 1)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = np.expand_dims(X_train, -1).astype("float32") / 255
X_test = np.expand_dims(X_test, -1).astype("float32") / 255

classifier = tf.keras.models.load_model("./models/mnist/classifier.keras")

# Load the generator
generator = Generator().eval()
generator.load_state_dict(torch.load('./models/mnist/netG_epoch_99.pth', map_location=torch.device('cpu')))

# Generate Images
generated_images = generator.generate_images(batch_size=1000)

# Get features
_, gen_features = classifier.predict(generated_images)

gen_rate = GeneralizationRate(classifier=classifier, X_train=X_train, X_test=X_test, delta=0.001).get(gen_features)
precision = Precision(classifier=classifier, X_train=X_train, X_test=X_test, delta=0.001).get(gen_features)
recall = Recall(classifier=classifier, X_train=X_train, X_test=X_test, delta=0.001).get(gen_features)
f1 = (3*recall*precision*gen_rate)/(recall+precision+gen_rate)
print("================")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Generalization Rate: {gen_rate}")

print((3*recall*precision*gen_rate)/(recall+precision+gen_rate))