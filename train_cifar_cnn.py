import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset_preprocessing import load_cifar10_data

class SimpleCifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # A small CNN architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        
        self.fc1   = nn.Linear(64 * 16 * 16, 256)
        self.fc2   = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))   # shape: [N, 32, 32, 32]
        x = F.relu(self.conv2(x))   # shape: [N, 64, 32, 32]
        x = self.pool(x)            # shape: [N, 64, 16, 16]
        
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))     # shape: [N, 256]
        x = self.fc2(x)             # shape: [N, 10]
        return x

def main():
    # 1. Load CIFAR-10 data
    train_loader, test_loader = load_cifar10_data(batch_size=64)
    
    # 2. Instantiate model, define loss and optimizer
    model = SimpleCifarCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20  # just for demo, can increase to 20 or more
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total
    
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Save model

    folder_path = "[...]" #your path
    file_name = "model_cnn_cifar10.pt"

    torch.save(model.state_dict(), folder_path + file_name)
    print("Saved CNN model parameters to model_cnn_cifar10.pt")

if __name__ == "__main__":
    main()


