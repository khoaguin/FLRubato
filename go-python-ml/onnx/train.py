import torch
import torch.nn as nn
import torch.onnx
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.layer(x)

def train_and_export_onnx(input_size, output_size, num_epochs=100):
    # Create dummy data for training
    X = torch.randn(100, input_size)
    y = torch.randn(100, output_size)
    
    # Initialize model
    model = SimpleModel(input_size, 64, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train the model
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Export the model to ONNX
    dummy_input = torch.randn(1, input_size)  # Example input
    torch.onnx.export(
        model,                 # model being run
        dummy_input,           # model input (or a tuple for multiple inputs)
        "model.onnx",          # where to save the model
        export_params=True,    # store the trained parameter weights inside the model file
        opset_version=12,      # the ONNX version to export the model to
        do_constant_folding=True,  # optimize model by folding constants
        input_names=['input'],     # the model's input names
        output_names=['output'],   # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print("Model exported to model.onnx")
    
    # Save a test input/output pair for verification
    test_input = np.random.randn(1, input_size).astype(np.float32)
    with torch.no_grad():
        test_output = model(torch.from_numpy(test_input)).numpy()
    
    np.save('test_input.npy', test_input)
    np.save('test_output.npy', test_output)
    print("Test data saved for verification")

if __name__ == "__main__":
    INPUT_SIZE = 10
    OUTPUT_SIZE = 5
    train_and_export_onnx(INPUT_SIZE, OUTPUT_SIZE)