import json
import torch
import torch.optim as optim
import torch.nn as nn
from model import IrisNet
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import random
from datetime import datetime

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load and prepare the dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Initialize model, criterion, and optimizer
model = IrisNet(config['input_dim'], config['hidden_dim'], config['output_dim'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

# Set up directories for saving results
run_dir = f"Runs/iris_dataset/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
os.makedirs(run_dir, exist_ok=True)
log_dir = os.path.join(run_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Training loop
for epoch in range(config['num_epochs']):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    # Log training metrics
    writer.add_scalar("Loss/train", loss.item(), epoch)

    # Periodic evaluation and logging
    if (epoch + 1) % 100 == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        accuracy = 100 * correct / total
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

# Save the model weights
model_path = os.path.join(run_dir, "results", "iris_model.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
writer.close()
