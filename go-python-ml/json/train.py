import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
from torchvision import datasets, transforms

class SimpleMNISTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMNISTModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def train_and_save_weights(
    input_size: int = 784, 
    hidden_size: int = 128, 
    output_size: int = 10, 
    num_epochs: int = 2
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = datasets.MNIST(
        "../../data/", download=True, train=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    
    # Initialize model
    model = SimpleMNISTModel(input_size, hidden_size, output_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training"):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader)}")

    # Extract weights and convert to nested Python list
    fc1 = model.fc1.weight.data.numpy()
    fc2 = model.fc2.weight.data.numpy()
    
    # Save weights and bias in JSON format
    weight_data = {
        'fc1': fc1.tolist(),
        'fc2': fc2.tolist(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size
    }
    
    with open('weights/model_weights.json', 'w') as f:
        json.dump(weight_data, f)
    
    print("Weights saved to model_weights.json")
    
    # Also save in NumPy format for verification
    np.save('weights/fc1.npy', fc1)
    np.save('weights/fc2.npy', fc2)

if __name__ == "__main__":
    INPUT_SIZE = 784
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10
    train_and_save_weights(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)