import torch
import torch.nn as nn

class FFNNModel(nn.Module):
    """
    Feed-Forward Neural Network for multivariate input, where each input sample
    is flattened to a vector of (sequence_length * num_features).
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)  # Pass through second hidden layer
        out = self.relu(out)  # Activation after second hidden layer
        out = self.fc3(out)  # Output layer
        return out

# Train function for FFNN
def train_ffnn_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Flatten inputs for FFNN
            inputs = inputs.view(inputs.size(0), -1)  # Flattening input for FFNN

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, -1, :])  # Get the last time step for targets

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')


class FFNNModel_uni(nn.Module):
    """
    Feed-Forward Neural Network for univariate input, where each input sample
    is of shape (sequence_length, 1), representing a single feature sequence.
    """
    def __init__(self, sequence_length, hidden_size, output_size=1):
        super(FFNNModel_uni, self).__init__()
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(sequence_length, hidden_size)  # Input layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        # Flatten the input from (sequence_length, 1) to (sequence_length) for the fully connected layer
        x = x.view(-1)  # Flatten (99, 1) to (99)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)  # Produces a single scalar output
        return out
