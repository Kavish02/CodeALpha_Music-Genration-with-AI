
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from model import MusicLSTM

# ---------------- CONFIGURATION ----------------

DATA_PATH = Path("preprocessed/training_sequences.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

SEQUENCE_LENGTH = 32
VOCAB_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("          AI MUSIC MODEL TRAINING")
print("=" * 50)

print("Training device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------- LOAD DATA ----------------

print("\nLoading training sequences...")

data = np.load(DATA_PATH)

X = torch.tensor(data["X"].astype(np.int64))
y = torch.tensor(data["y"].astype(np.int64))

dataset = TensorDataset(X, y)

print("Total samples:", len(dataset))
print("Input shape:", X.shape)
print("Target shape:", y.shape)

# 80% training, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print("Training samples:", train_size)
print("Validation samples:", val_size)

# ---------------- CREATE MODEL ----------------

model = MusicLSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=64,
    hidden_size=256,
    num_layers=2,
    dropout=0.2
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_loss = float("inf")

train_losses = []
val_losses = []

# ---------------- TRAINING ----------------

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for inputs, targets in progress:

        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(train_loader)

    # ---------------- VALIDATION ----------------

    model.eval()
    val_running_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            val_running_loss += loss.item()

    val_loss = val_running_loss / len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": VOCAB_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "embedding_dim": 64,
                "hidden_size": 256,
                "num_layers": 2,
                "best_val_loss": best_val_loss
            },
            MODEL_DIR / "best_music_model.pth"
        )

        print("Best model saved!")

# ---------------- SAVE TRAINING GRAPH ----------------

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Music Model Training")

plt.legend()
plt.grid(True)

plt.savefig(MODEL_DIR / "training_loss.png")
plt.close()

print("\n" + "=" * 50)
print("          TRAINING COMPLETED")
print("=" * 50)

print("Best validation loss:", best_val_loss)
print("Model saved in:", MODEL_DIR)
print("Training graph saved!")
