import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import onnx

# fixing random seed
random_seed = 42
torch.manual_seed(random_seed)

# Load and preprocess the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Define the neural network model
class IrisNet(nn.Module):
    def __init__(self):
        """
        Initializes the layers of the neural network.
        """
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # First fully connected layer (input: 4, output: 10)
        self.fc2 = nn.Linear(10, 10)  # Second fully connected layer (input: 10, output: 10)
        self.fc3 = nn.Linear(10, 3)  # Third fully connected layer (input: 10, output: 3)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Arguments:
        x -- input tensor

        Returns:
        Tensor containing the network's output after passing through the layers and activation functions.
        """
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the output of the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the output of the second layer
        x = self.fc3(x)  # Pass through the third layer (no activation)
        return x  # Return the final output


# Initialize the model, loss function, and optimizer
model = IrisNet()
criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Use Adam optimizer with a learning rate of 0.001

# Train the network
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Export the model to ONNX format
dummy_input = torch.randn(1, 4)  # Create a dummy input with the same shape as the network's input

# Export the trained model to the ONNX format for use in other frameworks or tools
torch.onnx.export(model, dummy_input, "iris_model.onnx", input_names=['input'], output_names=['output'])
