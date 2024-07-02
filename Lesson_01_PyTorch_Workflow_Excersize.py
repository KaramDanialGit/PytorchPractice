import torch
import torch.nn as nn
from pathlib import Path

# Constants
step = 0.1
weight = 0.3
bias = 0.9
epochs = 300


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    #     OR self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    #     OR return self.linear_layer(x)


x_data = torch.arange(1, 10, step).unsqueeze(dim=1)
y_data = weight * x_data + bias

train_idx = int(0.8 * len(x_data))
x_train = x_data[:train_idx]
x_test = x_data[train_idx:]
y_train = y_data[:train_idx]
y_test = y_data[train_idx:]

lin_model = LinearRegressionModel()
print(lin_model.state_dict())

loss_func = nn.L1Loss()
optimizer = torch.optim.SGD(params=lin_model.parameters(), lr=0.01)

for epoch in range(epochs):
    lin_model.train()

    y_pred = lin_model(x_train)
    loss = loss_func(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        lin_model.eval()
        with torch.inference_mode():
            test_pred = lin_model(x_test)
            test_loss = loss_func(test_pred, y_test)

        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

SAVE_PATH = Path("models")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = SAVE_PATH / MODEL_NAME
print(f"Saving model to: {MODEL_SAVE_PATH}")

torch.save(obj=lin_model.state_dict(), f=MODEL_SAVE_PATH)

# If we wanted to load a model:
# # Instantiate a fresh instance of LinearRegressionModelV2
# loaded_model_1 = LinearRegressionModelV2()
#
# # Load model state dict
# loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
