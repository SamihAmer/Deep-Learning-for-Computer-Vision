"""
HW#4 — Deep Learning for Computer Vision (EN525.733.8VL)
Autoencoder for MNIST Dimensionality Reduction
Due: March 5th, 2026

Assignment tasks:
  Q1-1. Train 3-layer MLP autoencoders on MNIST with varying bottleneck sizes
        (hidden units = 5, 10, 20, 30, 60, 120).
        Plot MSE learning curves (train / validation / test) vs. epoch.
        Evaluate reconstruction MSE specifically on digit "3".
  Q1-2. For 10 and 20 hidden units, visualize each encoder neuron's incoming
        weights as a 28×28 image (reveals what features each neuron detects).

Architecture (3-layer MLP, symmetric autoencoder):
  Layer L1 (Input)  : 784 units  ← flattened 28×28 MNIST image
  Layer L2 (Hidden) : N units    ← bottleneck / latent representation
  Layer L3 (Output) : 784 units  → reconstructed image

Loss function: Mean Squared Error (MSE)
Optimizer:     Adam
"""

import os
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — saves figures to disk
import matplotlib.pyplot as plt  # plotting library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SELECTION
# torch.cuda.is_available() returns True when a CUDA GPU is present.
# On your Mac this will be False (CPU); on your PC with NVIDIA GPU it will be
# True and training will automatically use the GPU for speed.
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD MNIST DATA
#
# MNIST is stored in IDX binary format:
#   Images file (IDX3): magic(4B) | num_images(4B) | rows(4B) | cols(4B) | pixels
#   Labels file (IDX1): magic(4B) | num_labels(4B) | labels
# All multi-byte integers are big-endian ('>I' in Python struct).
# Pixel values are uint8 in [0, 255]; we normalize to float32 in [0, 1].
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_images(filepath):
    """
    Read an MNIST IDX3-ubyte image file.

    Returns
    -------
    images : np.ndarray, shape (N, 784), dtype float32
        Each row is one flattened 28×28 image with pixel values in [0, 1].
    """
    with open(filepath, 'rb') as f:
        # Header: 4 unsigned 32-bit big-endian integers
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read all remaining bytes as pixel data
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    # Reshape to (N, 784) and scale pixels from [0,255] → [0,1]
    images = raw.reshape(num_images, rows * cols).astype(np.float32) / 255.0
    return images


def load_mnist_labels(filepath):
    """
    Read an MNIST IDX1-ubyte label file.

    Returns
    -------
    labels : np.ndarray, shape (N,), dtype int64
        Integer class labels in {0, 1, …, 9}.
    """
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    return raw.astype(np.int64)


# Build path to the MNIST archive relative to this script's location.
# Folder structure:  <script_dir>/archive/
# (HW_4.py lives inside the HW4 Autoencoder/ folder alongside the archive/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "archive")

train_images = load_mnist_images(os.path.join(DATA_DIR, "train-images.idx3-ubyte"))
train_labels = load_mnist_labels(os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))
test_images  = load_mnist_images(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte"))
test_labels  = load_mnist_labels(os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))

print(f"Loaded — Train: {train_images.shape}  Test: {test_images.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TRAIN / VALIDATION SPLIT
#
# The assignment says use 20% of the 60,000 training images for validation
# and 80% for training.  We shuffle first so every digit class is represented
# proportionally in both sets.
# ─────────────────────────────────────────────────────────────────────────────

VAL_RATIO = 0.20
N_total   = len(train_images)               # 60 000
N_val     = int(N_total * VAL_RATIO)        # 12 000
N_train   = N_total - N_val                 # 48 000

# Reproducible random shuffle
np.random.seed(42)
shuffle_idx  = np.random.permutation(N_total)
train_images = train_images[shuffle_idx]
train_labels = train_labels[shuffle_idx]

# First N_val samples → validation; rest → training
val_images   = train_images[:N_val]
val_labels   = train_labels[:N_val]
train_images = train_images[N_val:]
train_labels = train_labels[N_val:]

print(f"Split  — Train: {train_images.shape[0]}  Val: {val_images.shape[0]}  Test: {test_images.shape[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PYTORCH DATALOADERS
#
# PyTorch DataLoader handles batching and (optionally) shuffling for us.
# For an autoencoder the TARGET equals the INPUT — we want to reconstruct x.
# So TensorDataset(x, x) gives (input, target) pairs where both are the image.
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 256   # number of images processed per gradient update step


def make_loader(images, shuffle=True):
    """
    Wrap a numpy image array as a PyTorch DataLoader.

    The dataset yields (x, x) pairs because the autoencoder reconstructs
    its own input — there are no separate labels needed for training.
    """
    tensor  = torch.from_numpy(images)        # convert numpy → torch tensor
    dataset = TensorDataset(tensor, tensor)   # (input, target) = (x, x)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=(device.type == "cuda"))
    # pin_memory=True speeds up CPU→GPU transfers on CUDA machines


train_loader = make_loader(train_images, shuffle=True)   # shuffle every epoch
val_loader   = make_loader(val_images,   shuffle=False)  # order doesn't matter for eval
test_loader  = make_loader(test_images,  shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — AUTOENCODER MODEL
#
# A 3-layer MLP autoencoder has:
#   Encoder: Linear(784 → N) + Sigmoid activation
#   Decoder: Linear(N → 784) + Sigmoid activation
#
# Why Sigmoid?
#   • Hidden layer: compresses features into (0, 1) range, enforcing a compact
#     nonlinear representation in the latent space.
#   • Output layer: pixel values are in [0, 1] so Sigmoid is the natural output
#     activation — it prevents the network from predicting values outside that range.
#
# The encoder weight matrix W1 has shape (N, 784).
# Row i of W1 = the 784 weights converging onto hidden neuron i.
# Reshaping row i to 28×28 gives the "receptive field" of neuron i — what
# visual pattern it has learned to respond to (course notes pp. 78–79).
# ─────────────────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, hidden_units: int):
        """
        Parameters
        ----------
        hidden_units : int
            Number of neurons in the bottleneck (latent) layer.
            Smaller values = more compression but higher reconstruction error.
        """
        super(Autoencoder, self).__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        # Linear layer: y = W1 · x + b1,  W1 shape: (hidden_units, 784)
        # Sigmoid:       a = 1 / (1 + exp(-y)),  squashes output to (0, 1)
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_units),
            nn.Sigmoid()
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # Linear layer: y = W2 · a + b2,  W2 shape: (784, hidden_units)
        # Sigmoid:       output in (0, 1) matches normalized pixel range
        self.decoder = nn.Sequential(
            nn.Linear(hidden_units, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Full autoencoder pass: x → encoder → latent code → decoder → x̂

        Parameters
        ----------
        x : Tensor, shape (batch, 784)

        Returns
        -------
        recon : Tensor, shape (batch, 784)  — reconstructed images
        """
        latent = self.encoder(x)   # compress: 784 → hidden_units
        recon  = self.decoder(latent)  # reconstruct: hidden_units → 784
        return recon

    def get_encoder_weights(self):
        """
        Return the encoder's weight matrix as a numpy array.

        Shape: (hidden_units, 784)
        Row i contains the 784 weights converging to hidden neuron i.
        Reshape row i to (28, 28) to visualize it as an image.
        """
        # .data avoids autograd overhead; .cpu() moves from GPU if needed
        return self.encoder[0].weight.data.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse_on_loader(model, criterion, loader):
    """
    Compute the average MSE over every batch in a DataLoader.

    We use torch.no_grad() during evaluation to:
      1. Skip gradient computation (saves memory and time)
      2. Ensure model weights are not accidentally updated
    """
    total_loss  = 0.0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            recon    = model(x_batch)
            # sum of squared differences (reduction='sum' avoids batch-size bias)
            loss     = criterion(recon, x_batch)
            total_loss  += loss.item() * x_batch.size(0)
            total_count += x_batch.size(0)

    return total_loss / total_count   # average MSE across all samples


def compute_mse_for_digit(model, criterion, images, labels, digit=3):
    """
    Compute the average MSE only for samples of the specified digit class.

    This lets us answer "how well does the model reconstruct a '3'?"
    regardless of how it performs on other digits.

    Parameters
    ----------
    images : np.ndarray, shape (N, 784)
    labels : np.ndarray, shape (N,)
    digit  : int  — the digit class to evaluate (default = 3)
    """
    mask         = (labels == digit)           # boolean mask for this class
    digit_imgs   = torch.from_numpy(images[mask]).to(device)

    model.eval()
    with torch.no_grad():
        recon = model(digit_imgs)
        loss  = criterion(recon, digit_imgs)

    return loss.item()


def train_autoencoder(hidden_units, num_epochs=50, lr=1e-3):
    """
    Build, train, and return an autoencoder with `hidden_units` latent neurons.

    Training procedure:
      • Loss function: MSELoss — penalizes squared pixel-level reconstruction errors
      • Optimizer: Adam — adapts learning rates per parameter, converges reliably
      • Each epoch: one full pass through training data + evaluation on val & test

    Parameters
    ----------
    hidden_units : int   — bottleneck size
    num_epochs   : int   — number of full passes through the training set
    lr           : float — Adam learning rate

    Returns
    -------
    model        : trained Autoencoder
    train_losses : list[float] — per-epoch training MSE
    val_losses   : list[float] — per-epoch validation MSE
    test_losses  : list[float] — per-epoch test MSE
    """
    model     = Autoencoder(hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # MSELoss computes:  (1/n) * Σ (x_i - x̂_i)²  over all pixels in the batch
    criterion = nn.MSELoss()

    train_losses, val_losses, test_losses = [], [], []

    print(f"\n--- Training: {hidden_units} hidden units ---")

    for epoch in range(1, num_epochs + 1):

        # ── Training phase ───────────────────────────────────────────────────
        model.train()   # enables dropout/batchnorm if present (not used here,
                        # but good habit to always call model.train() before training)
        running_loss = 0.0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)   # move batch to GPU if available

            optimizer.zero_grad()           # reset gradients (they accumulate by default)
            recon = model(x_batch)          # forward pass: get reconstruction
            loss  = criterion(recon, x_batch)  # MSE between reconstruction and input
            loss.backward()                 # backpropagation: compute ∂loss/∂weights
            optimizer.step()               # gradient descent step: update weights

            # Accumulate weighted loss (multiply by batch size so we can average later)
            running_loss += loss.item() * x_batch.size(0)

        # Average loss over all training samples
        train_loss = running_loss / len(train_loader.dataset)

        # ── Evaluation phase ─────────────────────────────────────────────────
        # Evaluate on validation and test sets after each epoch
        val_loss  = compute_mse_on_loader(model, criterion, val_loader)
        test_loss = compute_mse_on_loader(model, criterion, test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: {train_loss:.5f} | "
                  f"Val: {val_loss:.5f} | "
                  f"Test: {test_loss:.5f}")

    return model, train_losses, val_losses, test_losses


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — RUN ALL EXPERIMENTS
#
# We train one autoencoder per hidden-unit count and store everything for
# downstream plotting and analysis.
# NOTE: 20 is added (not in the MSE-curve list of 5/10/30/60/120) because
# Q1-2 asks to visualize weights for 10 AND 20 hidden units.
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN_UNITS_LIST = [5, 10, 20, 30, 60, 120]  # bottleneck sizes to experiment with
NUM_EPOCHS        = 50                          # training epochs per model

criterion = nn.MSELoss()   # shared loss function for all evaluations

# Dictionaries to hold results indexed by hidden unit count
all_results = {}   # {h: (model, train_losses, val_losses, test_losses)}
digit3_mse  = {}   # {h: MSE on test images of digit "3"}

for h in HIDDEN_UNITS_LIST:
    model, train_l, val_l, test_l = train_autoencoder(
        hidden_units=h,
        num_epochs=NUM_EPOCHS,
        lr=1e-3
    )
    all_results[h] = (model, train_l, val_l, test_l)

    # Evaluate specifically on digit "3" from the TEST set
    # (the assignment asks us to compare reconstruction quality for digit 3)
    mse3 = compute_mse_for_digit(model, criterion, test_images, test_labels, digit=3)
    digit3_mse[h] = mse3
    print(f"  → Digit-3 Test MSE ({h} hidden units): {mse3:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — PLOT: Learning Curves (MSE vs Epoch)
#
# One subplot per hidden-unit configuration; each shows three curves:
#   blue  = training MSE
#   orange = validation MSE  (dashed)
#   green  = test MSE         (dotted)
#
# Comparing train vs. val/test curves reveals overfitting:
#   • If train MSE keeps dropping but val/test plateau → overfitting
#   • If all three curves decrease together → good generalization
# ─────────────────────────────────────────────────────────────────────────────

epochs = range(1, NUM_EPOCHS + 1)
n_plots = len(HIDDEN_UNITS_LIST)
n_cols  = 3
n_rows  = (n_plots + n_cols - 1) // n_cols   # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = np.array(axes).flatten()   # make 1D for easy indexing

for idx, h in enumerate(HIDDEN_UNITS_LIST):
    _, train_l, val_l, test_l = all_results[h]
    ax = axes[idx]

    ax.plot(epochs, train_l, color='steelblue',  linewidth=1.8, label='Train')
    ax.plot(epochs, val_l,   color='darkorange', linewidth=1.8,
            linestyle='--', label='Validation')
    ax.plot(epochs, test_l,  color='seagreen',   linewidth=1.8,
            linestyle=':',  label='Test')

    ax.set_title(f'Hidden units = {h}', fontsize=11)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Hide any unused subplot panels
for idx in range(n_plots, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('MLP Autoencoder — Learning Curves (MSE vs Epoch) on MNIST', fontsize=14)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150)
plt.close()
print("\nSaved: learning_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PLOT: Digit-3 MSE vs. Bottleneck Size (bar chart)
#
# This summarizes the final reconstruction quality for digit "3" across all
# bottleneck sizes.  More hidden units → more representational capacity →
# lower reconstruction error (up to a point).
# ─────────────────────────────────────────────────────────────────────────────

hs   = list(digit3_mse.keys())
mses = list(digit3_mse.values())

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar([str(h) for h in hs], mses, color='steelblue', edgecolor='black', alpha=0.85)

# Annotate each bar with its numeric MSE value
for bar, mse_val in zip(bars, mses):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mses) * 0.01,
            f'{mse_val:.5f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Number of Hidden Units (Bottleneck Size)', fontsize=11)
ax.set_ylabel('Test MSE — Digit "3"', fontsize=11)
ax.set_title('Reconstruction Error for Digit "3" vs. Bottleneck Size', fontsize=13)
ax.set_ylim(0, max(mses) * 1.15)
plt.tight_layout()
plt.savefig('digit3_mse_vs_hidden.png', dpi=150)
plt.close()
print("Saved: digit3_mse_vs_hidden.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — ENCODER WEIGHT VISUALIZATION (for 10 and 20 hidden units)
#
# The encoder's weight matrix W1 has shape (hidden_units, 784).
# Row i of W1 holds the 784 weights w_{1,i} … w_{784,i} that connect every
# input pixel to hidden neuron i.
# Reshaping that row to 28×28 and displaying it as a grayscale image shows
# the "template" or "filter" that neuron i has learned — pixels it responds
# strongly to will be bright; ignored pixels will be dark/neutral.
#
# With only 10 units the network must compress MNIST into 10 features, so
# each weight image tends to look like a broad, prototypical digit shape.
# With 20 units the features become more fine-grained.
# (See course notes pp. 78–79)
# ─────────────────────────────────────────────────────────────────────────────

for h in [10, 20]:
    model = all_results[h][0]
    W = model.get_encoder_weights()   # shape: (h, 784)

    # Normalize each weight image individually to [0, 1] for display.
    # (Global normalization is an alternative; per-neuron highlights each
    #  neuron's relative pattern more clearly.)
    W_min = W.min(axis=1, keepdims=True)   # shape: (h, 1)
    W_max = W.max(axis=1, keepdims=True)
    W_norm = (W - W_min) / (W_max - W_min + 1e-8)   # avoid division by zero

    # Arrange in a grid: up to 10 neurons per row
    n_cols_w = min(h, 10)
    n_rows_w = (h + n_cols_w - 1) // n_cols_w

    fig, axes = plt.subplots(n_rows_w, n_cols_w,
                             figsize=(n_cols_w * 1.8, n_rows_w * 1.8))
    axes = np.array(axes).flatten()

    for i in range(h):
        weight_img = W_norm[i].reshape(28, 28)   # 784-d vector → 28×28 image
        axes[i].imshow(weight_img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Neuron {i + 1}', fontsize=8)
        axes[i].axis('off')

    # Hide unused panels (only relevant if h < n_rows_w * n_cols_w)
    for i in range(h, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Encoder Weights Visualized as 28×28 Images  ({h} hidden units)',
                 fontsize=12)
    plt.tight_layout()
    fname = f'encoder_weights_{h}units.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — RECONSTRUCTION GALLERY: original vs. reconstructed digit "3"
#
# Pick one "3" from the test set and run it through every trained model.
# Displays the original alongside each reconstruction — you can visually see
# how more hidden units produce sharper, more accurate reconstructions.
# ─────────────────────────────────────────────────────────────────────────────

# Find the first digit-3 in the test set
digit3_idx = np.where(test_labels == 3)[0][0]
sample_img = torch.from_numpy(test_images[digit3_idx]).unsqueeze(0).to(device)
# unsqueeze(0) adds a batch dimension: shape (784,) → (1, 784)

n_models = len(HIDDEN_UNITS_LIST)
fig, axes = plt.subplots(1, n_models + 1, figsize=(3.0 * (n_models + 1), 3.2))

# Left panel: original image
axes[0].imshow(test_images[digit3_idx].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Original\n(Digit 3)', fontsize=10)
axes[0].axis('off')

# Remaining panels: one reconstruction per bottleneck size
for idx, h in enumerate(HIDDEN_UNITS_LIST):
    model = all_results[h][0]
    model.eval()
    with torch.no_grad():
        recon_np = model(sample_img).cpu().numpy().reshape(28, 28)

    axes[idx + 1].imshow(recon_np, cmap='gray', vmin=0, vmax=1)
    axes[idx + 1].set_title(f'h = {h}\nMSE = {digit3_mse[h]:.4f}', fontsize=9)
    axes[idx + 1].axis('off')

fig.suptitle('Digit "3" Reconstruction Quality vs. Bottleneck Size', fontsize=13)
plt.tight_layout()
plt.savefig('digit3_reconstructions.png', dpi=150)
plt.close()
print("Saved: digit3_reconstructions.png")

print("\n=== All tasks complete! ===")
print("Output files:")
print("  learning_curves.png         — MSE vs epoch for all bottleneck sizes")
print("  digit3_mse_vs_hidden.png    — bar chart of digit-3 reconstruction MSE")
print("  encoder_weights_10units.png — 10 encoder weight images (28×28 each)")
print("  encoder_weights_20units.png — 20 encoder weight images (28×28 each)")
print("  digit3_reconstructions.png  — original vs reconstructed digit 3")
