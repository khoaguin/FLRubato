import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Create synthetic training data
np.random.seed(42)
X_train = torch.FloatTensor(np.random.randn(1000, 10))
y_train = torch.FloatTensor((X_train.sum(axis=1) > 0).float()).reshape(-1, 1)

# Initialize model, loss function, and optimizer
model = SimpleNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model in TorchScript format
model.eval()
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, "simple_model.pt")