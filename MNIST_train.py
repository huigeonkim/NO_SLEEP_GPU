import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.utils
import torch.nn.init
import torch.nn as nn
import numpy as np
from tqdm import tqdm


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    devices = torch.cuda.device_count()
else:
    device = torch.device("cpu")
    devices = 1


# Set the hyperparameter
lr = learning_rate = 0.01
training_epochs = 1000000
batch_size = 1024

# Random Seed Fix
torch.manual_seed(777)

# Define the datasets
mnist_train = datasets.MNIST(root = 'Desktop/',
                       train = True,
                       transform = transforms.ToTensor(),
                       download = True)
mnist_test = datasets.MNIST(root = 'Desktop/',
                      train = False,
                      transform = transforms.ToTensor(),
                      download = True)

data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                          num_workers=8,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last = True)

# Define the architecture
class ComplexCNN(torch.nn.Module):

    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.keep_prob = 0.5

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L5 FC 1x1x256 inputs -> 1024 outputs
        self.fc1 = torch.nn.Linear(2 * 2 * 256, 1024, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer5 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        # L6 FC 1024 inputs -> 512 outputs
        self.fc2 = torch.nn.Linear(1024, 512, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.layer6 = torch.nn.Sequential(
            self.fc2,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        # L7 Final FC 512 inputs -> 10 outputs
        self.fc3 = torch.nn.Linear(512, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Flatten and print shape before FC layer
        out = out.view(out.size(0), -1)
        print("Flattened output shape:", out.shape)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc3(out)
        return out

model = ComplexCNN().to(device)
if devices > 1:
    model = nn.DataParallel(model)


criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_batch = len(data_loader)
print('총 배치의 수:', total_batch)

# Train
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in tqdm(data_loader, desc=f"Epoch {epoch+1}/{training_epochs}"):
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] Cost = {:>.9}'.format(epoch + 1, avg_cost))

    # Calculate accuracy for this epoch
    with torch.no_grad():
        X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
        Y_test = mnist_test.targets.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())



        



                        


