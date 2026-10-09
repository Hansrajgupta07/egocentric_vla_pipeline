import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import TinyPolicy

# ---------------------------------------------------------
# STEP 2: TRAIN BEHAVIORAL CLONING POLICY
# ---------------------------------------------------------

class ManipulationDataset(Dataset):
    def __init__(self, df):
        self.state_cols = ['hand_x', 'hand_y', 'hand_z', 'cube_x', 'cube_y', 'bowl_x', 'bowl_y']
        self.action_cols = ['d_x', 'd_y', 'd_z', 'd_gripper']
        self.X = df[self.state_cols].values.astype(np.float32)
        self.Y = df[self.action_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

def train_policy(csv_file="dataset.csv", epochs=100, batch_size=32, lr=1e-3):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run 1_extract_features.py first.")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded dataset: {len(df)} samples across {df['episode_id'].nunique()} episodes.")

    # Train / Val Split by episode
    unique_episodes = df['episode_id'].unique()
    np.random.seed(42)
    val_eps = set(np.random.choice(unique_episodes, size=max(1, int(len(unique_episodes) * 0.2)), replace=False))
    train_eps = [e for e in unique_episodes if e not in val_eps]

    train_df = df[df['episode_id'].isin(train_eps)]
    val_df = df[df['episode_id'].isin(val_eps)]

    print(f"Train set: {len(train_df)} samples ({len(train_eps)} episodes)")
    print(f"Validation set: {len(val_df)} samples ({len(val_eps)} episodes)")

    train_dataset = ManipulationDataset(train_df)
    val_dataset = ManipulationDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = TinyPolicy(input_dim=7, hidden_dim=128, output_dim=4).to(device)
    criterion_pos = nn.SmoothL1Loss()
    criterion_grip = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_pos_mse": []}
    best_val_loss = float("inf")

    print("\nStarting Policy Optimization...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            
            # Loss on 3D translation deltas + gripper state
            loss_pos = criterion_pos(preds[:, :3], y[:, :3])
            loss_grip = criterion_grip(preds[:, 3], y[:, 3])
            loss = loss_pos * 10.0 + loss_grip
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_pos_mse = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss_pos = criterion_pos(preds[:, :3], y[:, :3])
                loss_grip = criterion_grip(preds[:, 3], y[:, 3])
                loss = loss_pos * 10.0 + loss_grip
                val_loss += loss.item() * len(x)
                val_pos_mse += nn.MSELoss()(preds[:, :3], y[:, :3]).item() * len(x)

        val_loss /= len(val_dataset)
        val_pos_mse /= len(val_dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pos_mse"].append(val_pos_mse)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "policy.pt")

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Pos MSE: {val_pos_mse:.6f}")

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete! Best model checkpoint saved to 'policy.pt'.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_policy()
