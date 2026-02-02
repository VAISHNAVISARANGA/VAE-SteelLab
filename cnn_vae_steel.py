"""
CNN-VAE for Steel Microstructure Analysis
==========================================
Multi-task learning model that:
1. Reconstructs microstructure images (VAE)
2. Predicts process parameters
3. Predicts microstructural outputs
4. Predicts mechanical properties

Author: Steel Analysis System
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for the model"""
    
    # Data parameters
    IMAGE_SIZE = 128
    LATENT_DIM = 128
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # Training parameters
    EPOCHS = 150
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    
    # Loss weights
    LAMBDA_RECON = 1.0
    LAMBDA_KL = 0.0001
    LAMBDA_REGRESSION = 10.0
    
    # Early stopping
    PATIENCE = 20
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output features (15 total)
    # Process Parameters: 6 features
    PROCESS_PARAMS = ['Aust_Temp', 'Aust_Time', 'Cooling_Rate', 
                     'Quench_Temp', 'Temp_Temp', 'Temp_Time']
    
    # Microstructural Outputs: 5 features
    MICROSTRUCTURE = ['Martensite_Pct', 'Bainite_Pct', 'Ferrite_Pct', 
                      'Pearlite_Pct', 'Grain_Size']
    
    # Mechanical Properties: 4 features
    MECHANICAL = ['Hardness_HRC', 'Yield_Strength', 'Tensile_Strength', 'Elongation']
    
    # All regression targets
    ALL_TARGETS = PROCESS_PARAMS + MICROSTRUCTURE + MECHANICAL
    
    # Paths
    CSV_PATH = 'data_generator/data/steel_heat_treatment_data.csv'  # Adjusted to match your file name
    IMAGE_DIR = 'data_generator/data/microstructures'
    MODEL_SAVE_PATH = 'cnn_vae_steel.pth'
    SCALER_SAVE_PATH = 'target_scaler.pkl'

# ============================================================================
# DATASET CLASS
# ============================================================================

class SteelMicrostructureDataset(Dataset):
    """
    Custom Dataset for loading steel microstructure images and properties
    
    Args:
        csv_file (str): Path to CSV file
        image_dir (str): Directory containing microstructure images
        transform (callable): Optional transform to apply to images
        target_scaler (StandardScaler): Scaler for target values
        mode (str): 'train' or 'test' mode
    """
    
    def __init__(self, csv_file, image_dir, transform=None, target_scaler=None, mode='train'):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.target_scaler = target_scaler
        
        # Extract target columns
        self.targets = self.data[Config.ALL_TARGETS].values.astype(np.float32)
        
        # Fit scaler on training data
        if mode == 'train' and target_scaler is None:
            self.target_scaler = StandardScaler()
            self.targets_scaled = self.target_scaler.fit_transform(self.targets)
        elif target_scaler is not None:
            self.target_scaler = target_scaler
            self.targets_scaled = self.target_scaler.transform(self.targets)
        else:
            self.targets_scaled = self.targets
            
        print(f"[{mode.upper()}] Loaded {len(self.data)} samples")
        print(f"Target shape: {self.targets.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.data.iloc[idx]['Microstructure_Image']
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color=(128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get scaled targets
        targets = torch.FloatTensor(self.targets_scaled[idx])
        
        # Get original targets for reference
        targets_original = torch.FloatTensor(self.targets[idx])
        
        return {
            'image': image,
            'targets': targets,
            'targets_original': targets_original,
            'image_name': img_name
        }

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_train_transforms():
    """Data augmentation for training"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])

def get_test_transforms():
    """No augmentation for validation/test"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

# ============================================================================
# CNN-VAE ENCODER
# ============================================================================

class Encoder(nn.Module):
    """
    Convolutional Encoder for VAE
    
    Architecture:
    - 5 Convolutional blocks with BatchNorm and ReLU
    - Progressive channel increase: 3 -> 32 -> 64 -> 128 -> 256 -> 512
    - Outputs mu and log_var for latent space
    """
    
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = self._conv_block(3, 32)      # 128 -> 64
        self.conv2 = self._conv_block(32, 64)     # 64 -> 32
        self.conv3 = self._conv_block(64, 128)    # 32 -> 16
        self.conv4 = self._conv_block(128, 256)   # 16 -> 8
        self.conv5 = self._conv_block(256, 512)   # 8 -> 4
        
        # Calculate flattened size: 512 channels * 4 * 4 = 8192
        self.flatten_size = 512 * 4 * 4
        
        # Latent space parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
    def _conv_block(self, in_channels, out_channels):
        """Helper function to create a conv block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image [B, 3, 128, 128]
            
        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

# ============================================================================
# CNN-VAE DECODER
# ============================================================================

class Decoder(nn.Module):
    """
    Convolutional Decoder for VAE
    
    Architecture:
    - Mirrors encoder with transposed convolutions
    - Reconstructs image from latent vector
    """
    
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Project latent vector to feature map
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Transposed convolutional layers (mirror of encoder)
        self.deconv1 = self._deconv_block(512, 256)   # 4 -> 8
        self.deconv2 = self._deconv_block(256, 128)   # 8 -> 16
        self.deconv3 = self._deconv_block(128, 64)    # 16 -> 32
        self.deconv4 = self._deconv_block(64, 32)     # 32 -> 64
        
        # Final layer to get 3 channels
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def _deconv_block(self, in_channels, out_channels):
        """Helper function to create a deconv block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                             stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z):
        """
        Forward pass
        
        Args:
            z: Latent vector [B, latent_dim]
            
        Returns:
            recon: Reconstructed image [B, 3, 128, 128]
        """
        x = self.fc(z)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape to feature map
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        
        return x

# ============================================================================
# REGRESSION HEAD
# ============================================================================

class RegressionHead(nn.Module):
    """
    Multi-layer perceptron for property prediction
    
    Predicts 15 properties from latent vector:
    - 6 process parameters
    - 5 microstructural outputs
    - 4 mechanical properties
    """
    
    def __init__(self, latent_dim=128, output_dim=15):
        super(RegressionHead, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(64, output_dim)
        )
        
    def forward(self, z):
        """
        Forward pass
        
        Args:
            z: Latent vector [B, latent_dim]
            
        Returns:
            predictions: Property predictions [B, 15]
        """
        return self.network(z)

# ============================================================================
# COMPLETE CNN-VAE MODEL
# ============================================================================

class CNNVAE(nn.Module):
    """
    Complete CNN-VAE model with multi-task learning
    
    Combines:
    - Encoder (image -> latent space)
    - Decoder (latent space -> reconstructed image)
    - Regression head (latent space -> properties)
    """
    
    def __init__(self, latent_dim=128, num_properties=15):
        super(CNNVAE, self).__init__()
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.regression_head = RegressionHead(latent_dim, num_properties)
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE
        
        z = mu + sigma * epsilon
        where epsilon ~ N(0, 1)
        
        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
            
        Returns:
            z: Sampled latent vector [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image [B, 3, 128, 128]
            
        Returns:
            recon: Reconstructed image [B, 3, 128, 128]
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
            predictions: Property predictions [B, 15]
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decoder(z)
        
        # Predict properties
        predictions = self.regression_head(z)
        
        return recon, mu, logvar, predictions
    
    def encode(self, x):
        """Encode image to latent space (for inference)"""
        mu, logvar = self.encoder(x)
        return mu
    
    def decode(self, z):
        """Decode latent vector to image (for inference)"""
        return self.decoder(z)
    
    def predict(self, x):
        """Predict properties from image (for inference)"""
        mu, _ = self.encoder(x)
        predictions = self.regression_head(mu)
        return predictions

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class VAELoss(nn.Module):
    """
    Combined loss for CNN-VAE with multi-task learning
    
    Total Loss = λ_recon * Reconstruction Loss 
               + λ_kl * KL Divergence 
               + λ_reg * Regression Loss
    """
    
    def __init__(self, lambda_recon=1.0, lambda_kl=0.0001, lambda_regression=10.0):
        super(VAELoss, self).__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.lambda_regression = lambda_regression
        
        self.mse = nn.MSELoss()
        
    def reconstruction_loss(self, recon, x):
        """
        Reconstruction loss (MSE)
        
        Could also use BCE for binary images
        """
        return self.mse(recon, x)
    
    def kl_divergence(self, mu, logvar):
        """
        KL divergence between learned distribution and N(0, 1)
        
        KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
    
    def regression_loss(self, predictions, targets):
        """
        MSE loss for property predictions
        """
        return self.mse(predictions, targets)
    
    def forward(self, recon, x, mu, logvar, predictions, targets):
        """
        Compute total loss
        
        Args:
            recon: Reconstructed image
            x: Original image
            mu: Latent mean
            logvar: Latent log variance
            predictions: Predicted properties
            targets: Ground truth properties
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses for logging
        """
        recon_loss = self.reconstruction_loss(recon, x)
        kl_loss = self.kl_divergence(mu, logvar)
        reg_loss = self.regression_loss(predictions, targets)
        
        total_loss = (self.lambda_recon * recon_loss + 
                     self.lambda_kl * kl_loss + 
                     self.lambda_regression * reg_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'regression': reg_loss.item()
        }
        
        return total_loss, loss_dict

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    epoch_losses = {
        'total': 0.0,
        'recon': 0.0,
        'kl': 0.0,
        'regression': 0.0
    }
    
    pbar = tqdm(dataloader, desc='Training')
    
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward pass
        recon, mu, logvar, predictions = model(images)
        
        # Compute loss
        loss, loss_dict = criterion(recon, images, mu, logvar, predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses.keys():
            epoch_losses[key] += loss_dict[key]
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Average losses
    for key in epoch_losses.keys():
        epoch_losses[key] /= len(dataloader)
    
    return epoch_losses

def validate_epoch(model, dataloader, criterion, device, scaler=None):
    """Validate for one epoch"""
    model.eval()
    
    epoch_losses = {
        'total': 0.0,
        'recon': 0.0,
        'kl': 0.0,
        'regression': 0.0
    }
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['targets'].to(device)
            targets_original = batch['targets_original'].to(device)
            
            # Forward pass
            recon, mu, logvar, predictions = model(images)
            
            # Compute loss
            loss, loss_dict = criterion(recon, images, mu, logvar, predictions, targets)
            
            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]
            
            # Store predictions for metrics
            if scaler is not None:
                # Inverse transform predictions
                pred_np = predictions.cpu().numpy()
                pred_original = scaler.inverse_transform(pred_np)
                all_predictions.append(pred_original)
            else:
                all_predictions.append(predictions.cpu().numpy())
            
            all_targets.append(targets_original.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Average losses
    for key in epoch_losses.keys():
        epoch_losses[key] /= len(dataloader)
    
    # Compute metrics
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate R² and MAE for each property
    from sklearn.metrics import r2_score, mean_absolute_error
    
    metrics = {}
    for i, prop_name in enumerate(Config.ALL_TARGETS):
        r2 = r2_score(all_targets[:, i], all_predictions[:, i])
        mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        metrics[prop_name] = {'r2': r2, 'mae': mae}
    
    return epoch_losses, metrics

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(csv_path, image_dir, save_path='cnn_vae_steel.pth'):
    """
    Main training function
    
    Args:
        csv_path: Path to CSV file
        image_dir: Directory with microstructure images
        save_path: Path to save trained model
    """
    
    print("="*80)
    print("CNN-VAE for Steel Microstructure Analysis")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"Latent dimension: {Config.LATENT_DIM}")
    print(f"Number of targets: {len(Config.ALL_TARGETS)}")
    print()
    
    # Load and split data
    df = pd.read_csv(csv_path)
    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42
    )
    
    # Create datasets
    full_dataset = SteelMicrostructureDataset(
        csv_path, image_dir, 
        transform=get_train_transforms(),
        mode='train'
    )
    
    # Get the scaler from full dataset
    scaler = full_dataset.target_scaler
    
    # Save scaler
    import pickle
    with open(Config.SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {Config.SCALER_SAVE_PATH}")
    
    # Create train/val datasets with same scaler
    train_dataset = SteelMicrostructureDataset(
        csv_path, image_dir,
        transform=get_train_transforms(),
        target_scaler=scaler,
        mode='train'
    )
    train_dataset.data = df.iloc[train_idx].reset_index(drop=True)
    train_dataset.targets = df.iloc[train_idx][Config.ALL_TARGETS].values.astype(np.float32)
    train_dataset.targets_scaled = scaler.transform(train_dataset.targets)
    
    val_dataset = SteelMicrostructureDataset(
        csv_path, image_dir,
        transform=get_test_transforms(),
        target_scaler=scaler,
        mode='val'
    )
    val_dataset.data = df.iloc[val_idx].reset_index(drop=True)
    val_dataset.targets = df.iloc[val_idx][Config.ALL_TARGETS].values.astype(np.float32)
    val_dataset.targets_scaled = scaler.transform(val_dataset.targets)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    model = CNNVAE(
        latent_dim=Config.LATENT_DIM,
        num_properties=len(Config.ALL_TARGETS)
    ).to(Config.DEVICE)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss and optimizer
    criterion = VAELoss(
        lambda_recon=Config.LAMBDA_RECON,
        lambda_kl=Config.LAMBDA_KL,
        lambda_regression=Config.LAMBDA_REGRESSION
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2_avg': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 80)
        
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validate
        val_losses, val_metrics = validate_epoch(model, val_loader, criterion, Config.DEVICE, scaler)
        
        # Calculate average R²
        avg_r2 = np.mean([metrics['r2'] for metrics in val_metrics.values()])
        
        # Update learning rate
        scheduler.step(val_losses['total'])
        
        # Save history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['val_r2_avg'].append(avg_r2)
        
        # Print summary
        print(f"\nTrain Loss: {train_losses['total']:.4f} "
              f"(Recon: {train_losses['recon']:.4f}, "
              f"KL: {train_losses['kl']:.4f}, "
              f"Reg: {train_losses['regression']:.4f})")
        
        print(f"Val Loss: {val_losses['total']:.4f} "
              f"(Recon: {val_losses['recon']:.4f}, "
              f"KL: {val_losses['kl']:.4f}, "
              f"Reg: {val_losses['regression']:.4f})")
        
        print(f"Average R²: {avg_r2:.4f}")
        
        # Print top 5 best predicted properties
        r2_sorted = sorted(val_metrics.items(), key=lambda x: x[1]['r2'], reverse=True)
        print("\nTop 5 Best Predictions:")
        for prop, metrics in r2_sorted[:5]:
            print(f"  {prop:20s}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'metrics': val_metrics
            }, save_path)
            print(f"\n✓ Best model saved! (Val Loss: {best_val_loss:.4f})")
        
        # Early stopping check
        early_stopping(val_losses['total'])
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    return model, history, scaler

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def load_trained_model(model_path, scaler_path):
    """Load trained model and scaler"""
    
    # Load scaler
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    model = CNNVAE(
        latent_dim=Config.LATENT_DIM,
        num_properties=len(Config.ALL_TARGETS)
    ).to(Config.DEVICE)
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Trained for {checkpoint['epoch']+1} epochs")
    print(f"✓ Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, scaler

def predict_from_image(image_path, model, scaler, show_image=False):
    """
    Predict properties from a single microstructure image
    
    Args:
        image_path: Path to microstructure image
        model: Trained CNN-VAE model
        scaler: StandardScaler for targets
        show_image: Whether to display the image
        
    Returns:
        results: Dictionary with predictions organized by category
    """
    
    # Load and preprocess image
    transform = get_test_transforms()
    image = Image.open(image_path).convert('RGB')
    
    if show_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title('Input Microstructure')
        plt.axis('off')
        plt.show()
    
    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions_scaled = model.predict(image_tensor)
        
        # Inverse transform
        predictions = scaler.inverse_transform(predictions_scaled.cpu().numpy())
    
    # Organize results
    results = {
        'Process Parameters': {},
        'Microstructure': {},
        'Mechanical Properties': {}
    }
    
    # Process parameters
    for i, param in enumerate(Config.PROCESS_PARAMS):
        results['Process Parameters'][param] = float(predictions[0, i])
    
    # Microstructure
    offset = len(Config.PROCESS_PARAMS)
    for i, param in enumerate(Config.MICROSTRUCTURE):
        results['Microstructure'][param] = float(predictions[0, offset + i])
    
    # Mechanical properties
    offset = len(Config.PROCESS_PARAMS) + len(Config.MICROSTRUCTURE)
    for i, param in enumerate(Config.MECHANICAL):
        results['Mechanical Properties'][param] = float(predictions[0, offset + i])
    
    return results

def print_predictions(results):
    """Pretty print prediction results"""
    
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    # Process Parameters
    print("\n📊 PROCESS PARAMETERS:")
    print("-" * 80)
    for param, value in results['Process Parameters'].items():
        if 'Temp' in param and 'Time' not in param:
            print(f"  {param:25s}: {value:8.1f} °C")
        elif 'Time' in param:
            print(f"  {param:25s}: {value:8.1f} min")
        elif 'Cooling_Rate' in param:
            print(f"  {param:25s}: {value:8.2f} °C/s")
        else:
            print(f"  {param:25s}: {value:8.2f}")
    
    # Microstructure
    print("\n🔬 MICROSTRUCTURE:")
    print("-" * 80)
    for param, value in results['Microstructure'].items():
        if 'Pct' in param:
            print(f"  {param:25s}: {value:8.2f} %")
        else:
            print(f"  {param:25s}: {value:8.2f} μm")
    
    # Check phase percentages sum
    phase_sum = sum([v for k, v in results['Microstructure'].items() if 'Pct' in k])
    print(f"  {'Phase Total':25s}: {phase_sum:8.2f} % (should be ~100%)")
    
    # Mechanical Properties
    print("\n⚙️  MECHANICAL PROPERTIES:")
    print("-" * 80)
    for param, value in results['Mechanical Properties'].items():
        if 'Hardness' in param:
            print(f"  {param:25s}: {value:8.1f} HRC")
        elif 'Strength' in param:
            print(f"  {param:25s}: {value:8.0f} MPa")
        elif 'Elongation' in param:
            print(f"  {param:25s}: {value:8.1f} %")
    
    print("\n" + "="*80)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_reconstruction(model, dataloader, num_samples=5, save_path=None):
    """Visualize original vs reconstructed images"""
    
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    images = batch['image'][:num_samples].to(Config.DEVICE)
    
    with torch.no_grad():
        recon, _, _, _ = model(images)
    
    # Move to CPU and convert to numpy
    images = images.cpu()
    recon = recon.cpu()
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Original
        img = images[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed
        rec = recon[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(rec)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Reconstruction visualization saved to {save_path}")
    
    plt.show()

def plot_training_history(history, save_path=None):
    """Plot training history"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R² plot
    axes[1].plot(history['val_r2_avg'], label='Avg R²', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R²', fontsize=12)
    axes[1].set_title('Validation R² Score', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to {save_path}")
    
    plt.show()

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    
    # Train the model
    print("\n🚀 Starting Model Training...\n")
    
    model, history, scaler = train_model(
        csv_path=Config.CSV_PATH,
        image_dir=Config.IMAGE_DIR,
        save_path=Config.MODEL_SAVE_PATH
    )
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_training_history(history, save_path='training_history.png')
    
    # Visualize reconstructions
    print("\n🖼️  Visualizing image reconstructions...")
    val_dataset = SteelMicrostructureDataset(
        Config.CSV_PATH, Config.IMAGE_DIR,
        transform=get_test_transforms(),
        target_scaler=scaler,
        mode='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    visualize_reconstruction(model, val_loader, num_samples=5, 
                           save_path='reconstructions.png')
    
    print("\n✅ Training complete! Model saved to", Config.MODEL_SAVE_PATH)
    print("✅ Scaler saved to", Config.SCALER_SAVE_PATH)