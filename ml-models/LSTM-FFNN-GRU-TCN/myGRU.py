import torch
import torch.nn as nn
import torch.optim as optim

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU layer instead of LSTM
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state for the first input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through GRU
        out, _ = self.gru(x, h0)
        
        # Fully connected layer to get the output (use the last time step's output)
        out = self.fc(out[:, -1, :])  # Many-to-one prediction (last GRU output)
        return out

def train_gru_model(model, train_loader, criterion, optimizer, num_epochs=2):
    """
    Train the GRU model.

    :param model: nn.Module, the GRU model
    :param train_loader: DataLoader, training data
    :param criterion: loss function
    :param optimizer: optimizer
    :param num_epochs: int, number of epochs to train
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, -1, :])  # Compare with the last time step
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
