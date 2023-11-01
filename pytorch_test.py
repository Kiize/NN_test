import torch
import torch.nn as nn
from torch.optim import SGD
import time
import matplotlib.pyplot as plt

start = time.time()

# Initial values.

x = torch.tensor(
    [
        [6,2],[5,2],[1,3],[7,6]
    ]
).float()

y = torch.tensor(
    [1,5,2,5]
).float()


# Model.

class MyNeuralNet(nn.Module):
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

class MyNeuralNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2, 80)
        self.Matrix2 = nn.Linear(80, 80)
        self.Matrix3 = nn.Linear(80, 1)
        self.R = nn.ReLU()

    def forward(self, x):
        """
        x: vector of inputs
        """
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()

def train_model(x, y, f, n_epochs=50):
    opt = SGD(f.parameters(), lr = 0.001)
    L = nn.MSELoss()
    losses = []

    for _ in range(n_epochs):
        opt.zero_grad() #flush previous epoch's gradient
        loss_value = L(f(x), y) #compute Loss
        loss_value.backward() #compute gradient
        opt.step() #perform iteration using gradient
        losses.append(loss_value.item())
    return f, losses

f2 = MyNeuralNet2()
f2, losses = train_model(x, y, f2)

plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')

print(f2(x))

# Time.

print(f"elapsed time: {(time.time() - start):.1f} seconds\n")

plt.show()