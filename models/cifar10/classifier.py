import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.models import resnet50

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Download test data from open datasets.
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
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

# Example instantiation of the model
model = ResNet50CIFAR10(num_classes=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

epochs = 1000
prev_test_loss = 1000
early_stopping_counter = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test_loss = test(test_dataloader, model, loss_fn)
    if test_loss > prev_test_loss:
        early_stopping_counter += 1
        print(early_stopping_counter)
    else:
        early_stopping_counter = 0
    if early_stopping_counter >= 3:
        print("Early stopping")
        break
    prev_test_loss = test_loss
print("Done!")

torch.save(model.state_dict(), "CIFAR10-Classifier.pth")
print("Saved PyTorch Model State to model.pth")