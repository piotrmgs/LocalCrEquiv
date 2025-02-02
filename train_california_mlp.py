import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset_preprocessing import load_california_housing_data

# 1. Define an MLP Model for California Housing
class MLPRegression(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        # Simple architecture: 2 hidden layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    # 2. Load data
    train_loader, test_loader = load_california_housing_data(batch_size=64)
    # input_dim=8 means 8 features in the dataset

    # 3. Instantiate model, define loss and optimizer
    model = MLPRegression(input_dim=8, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. Training loop
    epochs = 30  # you can increase to 30, but let's keep it short for demo
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Average loss per epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

    # 5. Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print(f"Final Test MSE: {test_loss:.4f}")

    # 6. Save the model state
    folder_path = "[...]" #your path
    file_name = "model_mlp_california.pt"

    torch.save(model.state_dict(), folder_path + file_name)
    print("Saved MLP model parameters to model_mlp_california.pt")

if __name__ == "__main__":
    main()
