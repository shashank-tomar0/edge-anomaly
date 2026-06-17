import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from models.autoencoder import AnomalyAutoencoder
import os

def train():
    print("Loading data...")
    data_path = 'data/normal_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run simulate_sensor_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    # Convert dataframe to numpy array then to tensor
    # Assuming each row is a sequence of length 64
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    
    dataset = TensorDataset(data_tensor, data_tensor) # Autoencoder targets itself
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    sequence_length = 64
    model = AnomalyAutoencoder(sequence_length)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print(f"Training model on normal data for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print("Model saved to autoencoder_model.pth")

if __name__ == "__main__":
    train()
