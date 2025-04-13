import torch
import torch.nn as nn
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset


class SimpleMNISTModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleMNISTModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def load_model_from_json(weights_path: Path) -> SimpleMNISTModel:
    # Load weights from JSON
    with open(weights_path, "r") as f:
        weight_data = json.load(f)

    # Create model with same architecture
    model = SimpleMNISTModel()

    # Convert weights back to tensors and load into model
    fc1_weights = torch.tensor(weight_data["fc1"])
    fc2_weights = torch.tensor(weight_data["fc2"])

    model.fc1.weight.data = fc1_weights
    model.fc2.weight.data = fc2_weights

    return model


def evaluate_model(weights_path, test_set: Dataset, device="cpu"):
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=2
    )
    model = load_model_from_json(weights_path)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy


def train_and_save_weights(
    train_set: Dataset, num_epochs: int = 2, weights_path: Path = None
):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=2
    )

    # Initialize model
    model = SimpleMNISTModel()
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
        "fc1": fc1.tolist(),
        "fc2": fc2.tolist(),
    }

    with open(weights_path, "w") as f:
        json.dump(weight_data, f)

    print(f"Weights saved to {weights_path}")

    # Also save in NumPy format for verification
    # np.save('weights/fc1.npy', fc1)
    # np.save('weights/fc2.npy', fc2)


# if __name__ == "__main__":
#     torch.manual_seed(42)
#     weights_path = Path("weights/model_weights_exclude_469.json")
#     train_set, test_set = load_mnist_data()
#     # train_part = exclude_digits(train_set, excluded_digits=[1, 3, 7])
#     # train_part = exclude_digits(train_set, excluded_digits=[2, 5, 8])
#     train_part = exclude_digits(train_set, excluded_digits=[4, 6, 9])
#     testset_137 = include_digits(test_set, [1, 3, 7])
#     testset_258 = include_digits(test_set, [2, 5, 8])
#     testset_469 = include_digits(test_set, [4, 6, 9])
#     total_length = len(train_set)
#     train_and_save_weights(train_part, 3, weights_path)
#     evaluate_model(weights_path, testset_137)
#     evaluate_model(weights_path, testset_258)
#     evaluate_model(weights_path, testset_469)
