import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np 
import time
import matplotlib.pyplot as plt

start = time.time()

# Initial values.
"""
x = torch.tensor(
    [
        [6,2],[5,2],[1,3],[7,6]
    ]
).float()

y = torch.tensor(
    [1,5,2,5]
).float()
"""

# Model.

class MyNeuralNet_test(nn.Module):
    def __init__(self):
        """
        M1 takes a 2xN vector and returns an 8xN matrix 
        """
        super().__init__()
        self.Matrix1 = nn.Linear(2, 8, bias=False)
        self.Matrix2 = nn.Linear(8, 1, bias=False)

    def forward(self, x):
        """
        x: vector of inputs
        """
        x = self.Matrix1(x)
        x = self.Matrix2(x)
        return x.squeeze()

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50, 10)
        self.R = nn.ReLU()

    def forward(self, x):
        """
        x: vector of inputs
        """
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()

def train_model(dl, f, n_epochs=20):
    opt = SGD(f.parameters(), lr = 0.01)
    L = nn.CrossEntropyLoss()

    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)

        for i, (x, y) in enumerate(dl):
            opt.zero_grad() #flush previous epoch's gradient
            loss_value = L(f(x), y) #compute Loss
            loss_value.backward() #compute gradient
            opt.step() #perform iteration using gradient

            epochs.append(epoch+i/N)
            losses.append(loss_value.item())

    return np.array(epochs), np.array(losses)

class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255. # image intensity from 0 to 1
        self.y = F.one_hot(self.y, num_classes=10).to(float) # One-Hot encoder.

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]

train_ds = CTDataset('/home/viper/Scaricati/MNIST/processed/training.pt')
test_ds = CTDataset('/home/viper/Scaricati/MNIST/processed/test.pt')

train_dl = DataLoader(train_ds, batch_size=5)


f = MyNeuralNet()
epoch_data, loss_data = train_model(train_dl, f)

xs, ys = train_ds[0:2000]
yhats = f(xs).argmax(axis=1)

# Plot
fig, ax = plt.subplots(10,4, figsize = (10, 15))

for i in range(40):
    plt.subplot(10, 4, i+1)
    plt.imshow(xs[i])
    plt.title(f'Predicted Digit: {yhats[i]}')
fig.tight_layout()

# Time.

print(f"elapsed time: {(time.time() - start):.1f} seconds\n")

plt.show()
