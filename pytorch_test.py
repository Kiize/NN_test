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

f = MyNeuralNet()
L = nn.MSELoss()

opt = SGD(f.parameters(), lr = 0.001)
losses = []

for _ in range(50):
    opt.zero_grad() #flush previous epoch's gradient
    loss_value = L(f(x), y) #compute Loss
    loss_value.backward() #compute gradient
    opt.step() #perform iteration using gradient
    losses.append(loss_value.item())

plt.plot(losses)
plt.xlabel('Loss')
plt.ylabel('Epochs')

# Time.

print(f"elapsed time: {(time.time() - start):.1f} seconds\n")

plt.show()