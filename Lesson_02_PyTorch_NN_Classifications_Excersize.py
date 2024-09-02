# Import all necessary libraries

import torch
from torch import nn
import numpy as np
from sklearn.datasets import make_moons
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

RANDOM_SEED = 42
samples = 1000

# Create binary data set
x, y = make_moons(n_samples=samples, noise=0.2, random_state=RANDOM_SEED)

# Turn data into DataFrame
moons = pd.DataFrame({
    "X1": x[:,0],
    "X2": x[:,1],
    "Label": y
})

# Plot the moons
plt.scatter(
    x=x[:, 0],
    y=x[:, 1],
    c=y
)

# Turn tensors to float
X = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split train and test data
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED
)

class MoonModelV0(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(in_features=2, out_features=10)
    self.layer2 = nn.Linear(in_features=10, out_features=10)
    self.layer3 = nn.Linear(in_features=10, out_features=1)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))\

moon_model = MoonModelV0()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(moon_model.parameters(), lr=0.1)

acc_fn = Accuracy(task="multiclass", num_classes=2)
acc_fn

# Train model and see what it's outputting

torch.manual_seed(RANDOM_SEED)
epochs = 100

for epoch in range(epochs):
    moon_model.train()

    y_logits = moon_model(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    moon_model.eval()

    with torch.inference_mode():
        test_logits = moon_model(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = acc_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.5f}, Test Accuracy: {test_acc:.5f}")


def plot_decision_boundary(model, x, y):
    model.to("cpu")
    X, y = x.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot decision boundaries for training and test sets
plot_decision_boundary(moon_model, x_train, y_train)
plot_decision_boundary(moon_model, x_test, y_test)