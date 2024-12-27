import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, num_class):
        super(MLP, self).__init__()

        self.perceptron = nn.Sequential(
            nn.Linear(input_size, 1024), #1024 bobot + 1 bias
            nn.ReLU(),
            nn.Linear(1024, 128), #128 bobot + 1 bias
            nn.ReLU(),
            nn.Linear(128, 64), #64 bobot + 1 bias
            nn.ReLU(),
            nn.Linear(64, 32), #32 bobot + 1 bias
            nn.ReLU(),
            nn.Linear(32, num_class)
        )
    
    def forward(self, x):
        x = self.perceptron(x)
        return x
'''
# testing dengan data dummy
if __name__=='__main__':
    input_tensor = torch.randn(1, 8)
    model = MLP(input_size=8, num_class=5)
    output = model(input_tensor)
    print("Output Shape:", output.shape)
'''