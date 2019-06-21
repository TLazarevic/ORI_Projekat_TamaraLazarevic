import torch.nn.functional as F
from torch import nn


class NeuralNet:

    def __init__(self):
        n_in, n_h, n_out, batch_size = 400, 5, 64, 10 #layer size and batch size
        x = torch.randn(batch_size, n_in)
        y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
        model = nn.Sequential(nn.Linear(n_in, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h, n_out),
                      nn.Sigmoid())
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def train(self):
        for epoch in range(50):
            # Forward Propagation
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y)
            print('epoch: ', epoch, ' loss: ', loss.item())
            # Zero the gradients
            optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()

