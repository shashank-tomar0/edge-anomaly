import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
from models.autoencoder import AnomalyAutoencoder
import config

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_mses = []
    with torch.no_grad():
        for batch_features, _ in dataloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            total_loss += loss.item()
            
            # Calculate sample-wise MSE
            batch_mse = torch.mean((batch_features - outputs) ** 2, dim=1).cpu().numpy()
            all_mses.extend(batch_mse)
            
    return total_loss / len(dataloader), np.array(all_mses)

def train_pipeline():
    print("Loading configuration...")
    print(f"Data Path: {config.NORMAL_DATA_PATH}")
    
    if not os.path.exists(config.NORMAL_DATA_PATH):
        print(f"Error: {config.NORMAL_DATA_PATH} not found. Run generate-data command first.")
        return
        
    df = pd.read_csv(config.NORMAL_DATA_PATH)
    data = df.values.astype(np.float32)
    
    # Shuffle and split into Train and Validation sets
    np.random.seed(42)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    split_idx = int(len(data) * config.TRAIN_SPLIT)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = torch.tensor(data[train_indices])
    val_data = torch.tensor(data[val_indices])
    
    train_dataset = TensorDataset(train_data, train_data)
    val_dataset = TensorDataset(val_data, val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Dataset summary: {len(train_data)} train samples, {len(val_data)} validation samples.")
    
    # Initialize model
    model = AnomalyAutoencoder(sequence_length=config.SEQUENCE_LENGTH, latent_dim=config.LATENT_DIM)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training Loop
    epochs = config.EPOCHS
    best_val_loss = float('inf')
    
    print(f"Training Anomaly Autoencoder for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_features, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        train_loss = total_train_loss / len(train_loader)
        val_loss, val_mses = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
    # Save the model
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    
    # Calculate threshold based on validation set MSEs
    # Re-run validation set in eval mode to get final MSEs
    _, val_mses = evaluate_model(model, val_loader, criterion)
    threshold = float(np.percentile(val_mses, config.THRESHOLD_PERCENTILE))
    print(f"Calculated anomaly threshold ({config.THRESHOLD_PERCENTILE}th percentile): {threshold:.5f}")
    
    # Evaluate on anomalous data (test set)
    metrics = {}
    if os.path.exists(config.ANOMALOUS_DATA_PATH):
        print("Evaluating model performance on anomalous dataset...")
        anom_df = pd.read_csv(config.ANOMALOUS_DATA_PATH)
        anom_data = torch.tensor(anom_df.values.astype(np.float32))
        anom_dataset = TensorDataset(anom_data, anom_data)
        anom_loader = DataLoader(anom_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        _, anom_mses = evaluate_model(model, anom_loader, criterion)
        
        # Validation set acts as normal test set
        # Anomalous set acts as anomalous test set
        tp = np.sum(anom_mses >= threshold)
        fn = np.sum(anom_mses < threshold)
        fp = np.sum(val_mses >= threshold)
        tn = np.sum(val_mses < threshold)
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        print("\n--- Evaluation Metrics ---")
        print(f"True Positives (Detected Anomalies): {tp}/{len(anom_mses)}")
        print(f"False Negatives (Missed Anomalies): {fn}/{len(anom_mses)}")
        print(f"False Positives (Normal flagged as Anomaly): {fp}/{len(val_mses)}")
        print(f"True Negatives (Normal classified correctly): {tn}/{len(val_mses)}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        metrics = {
            "test_anomalous_samples": len(anom_mses),
            "test_normal_samples": len(val_mses),
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
    # Save metadata
    metadata = {
        "sequence_length": config.SEQUENCE_LENGTH,
        "latent_dim": config.LATENT_DIM,
        "threshold": threshold,
        "threshold_percentile": config.THRESHOLD_PERCENTILE,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "metrics": metrics
    }
    
    with open(config.METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {config.METADATA_PATH}")

if __name__ == "__main__":
    train_pipeline()
