import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from scipy.stats.mstats import winsorize
from scipy.optimize import minimize
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table
from joblib import Parallel, delayed, load, dump
from pathlib import Path
try:
    import xgboost as xgb
except ImportError:
    xgb = None


# Initialize Console
console = Console()

# --- UTILS ---

def winsorize_series(s: pd.Series, p=0.01) -> pd.Series:
    """
    Winsorize a pandas Series.
    
    Args:
        s (pd.Series): Input time series data.
        p (float): Winsorization proportion, default 0.01 (1%). Clips p from both ends.
        
    Returns:
        pd.Series: Processed Series with extreme values replaced by quantiles.
    """
    mask = s.notna()
    x = s[mask].to_numpy()
    if len(x) == 0:
        return s
    x_w = winsorize(x, limits=(p, p))   # clip 1%
    out = s.copy()
    out.loc[mask] = np.asarray(x_w)
    return out

def _rolling_sum(x, W):
    x = np.asarray(x, dtype=float)
    cs = np.cumsum(np.pad(np.nan_to_num(x), (1, 0)))   # length T+1
    out_valid = cs[W:] - cs[:-W]                        # length T-W+1
    out = np.empty_like(x)
    out[:W-1] = np.nan
    out[W-1:] = out_valid
    return out

def generate_features_vectorized(ask_price_arry, bid_price_arry,
                                 ask_size_arry, bid_size_arry,
                                 ask_nc_arry, bid_nc_arry):
    """
    Vectorized generation of high-frequency price-volume features.
    
    Args:
        ask_price_arry (np.array): Ask price matrix (T, Levels)
        bid_price_arry (np.array): Bid price matrix (T, Levels)
        ask_size_arry (np.array): Ask size matrix (T, Levels)
        bid_size_arry (np.array): Bid size matrix (T, Levels)
        ask_nc_arry (np.array): Ask order count matrix (T, Levels)
        bid_nc_arry (np.array): Bid order count matrix (T, Levels)
        
    Returns:
        pd.DataFrame: DataFrame containing generated features (float32).
        Features include: mid-price momentum, imbalance, order flow, cumulative attack, spoofing proxy, etc.
    """
    console = Console()
    features = {}

    # Force input to float32 to reduce memory usage
    ask_price_arry = ask_price_arry.astype(np.float32)
    bid_price_arry = bid_price_arry.astype(np.float32)
    ask_size_arry = ask_size_arry.astype(np.float32)
    bid_size_arry = bid_size_arry.astype(np.float32)
    ask_nc_arry = ask_nc_arry.astype(np.float32)
    bid_nc_arry = bid_nc_arry.astype(np.float32)

    ask1_p = ask_price_arry[:, 0]
    bid1_p = bid_price_arry[:, 0]
    ask1_sz = ask_size_arry[:, 0]
    bid1_sz = bid_size_arry[:, 0]
    ask1_nc = ask_nc_arry[:, 0]
    bid1_nc = bid_nc_arry[:, 0]

    total_features = 19
    with Progress() as progress:
        task = progress.add_task("[cyan]Generating features...", total=total_features)

        # mid_price and spread - ensure float32
        mid_price = (ask1_p + bid1_p) * np.float32(0.5)
        progress.update(task, advance=1)
        
        ask_sz_sum = ask_size_arry.sum(axis=1, dtype=np.float32)
        bid_sz_sum = bid_size_arry.sum(axis=1, dtype=np.float32)
        tot_sz_sum = ask_sz_sum + bid_sz_sum

        ask_pq_sum = np.einsum('ij,ij->i', ask_price_arry, ask_size_arry, dtype=np.float32)
        bid_pq_sum = np.einsum('ij,ij->i', bid_price_arry, bid_size_arry, dtype=np.float32)
        progress.update(task, advance=1)

        # mid_price_momentum - ensure float32
        for n in (30, 60, 120):
            mom = (mid_price - np.roll(mid_price, n)).astype(np.float32)
            mom[:n] = np.nan
            features[f'mid_price_momentum_{n}'] = mom
            progress.update(task, advance=1)

        features['imbalance'] = ((ask1_sz - bid1_sz) / (ask1_sz + bid1_sz + np.float32(1e-8))).astype(np.float32)
        features['nc_imbalance'] = ((ask1_nc - bid1_nc) / (ask1_nc + bid1_nc + np.float32(1e-8))).astype(np.float32)
        progress.update(task, advance=1)

        features['price_weighted_imbalance_delta'] = ((ask_pq_sum - bid_pq_sum) / (ask_pq_sum + bid_pq_sum + np.float32(1e-8))).astype(np.float32)
        features['price_weighted_imbalance'] = (ask_pq_sum / (ask_pq_sum + bid_pq_sum + np.float32(1e-8))).astype(np.float32)
        progress.update(task, advance=1)

        features['opt_imbalance_delta'] = ((ask1_sz - bid1_sz) / (ask1_sz + bid1_sz + np.float32(1e-8))).astype(np.float32)
        features['total_imbalance_delta'] = ((ask_sz_sum - bid_sz_sum) / (tot_sz_sum + np.float32(1e-8))).astype(np.float32)
        progress.update(task, advance=1)

        features['opt_imbalance'] = (ask1_sz / (ask1_sz + bid1_sz + np.float32(1e-8))).astype(np.float32)
        features['total_imbalance'] = (ask_sz_sum / (tot_sz_sum + np.float32(1e-8))).astype(np.float32)
        progress.update(task, advance=1)

        W = 60

        bid_delta_1 = np.diff(bid1_sz, prepend=bid1_sz[0]).astype(np.float32)
        ask_delta_1 = np.diff(ask1_sz, prepend=ask1_sz[0]).astype(np.float32)
        net_order_flow = (bid_delta_1 - ask_delta_1).astype(np.float32)
        net_order_flow[0] = np.float32(0.0)
        features['net_order_flow'] = net_order_flow
        features['cumsum_net_flow'] = _rolling_sum(net_order_flow, W).astype(np.float32)
        progress.update(task, advance=1)

        bid_delta_all = np.diff(bid_sz_sum, prepend=bid_sz_sum[0]).astype(np.float32)
        ask_delta_all = np.diff(ask_sz_sum, prepend=ask_sz_sum[0]).astype(np.float32)
        net_order_flow_all = (bid_delta_all - ask_delta_all).astype(np.float32)
        net_order_flow_all[0] = np.float32(0.0)
        features['net_order_flow_all'] = net_order_flow_all
        features['cumsum_net_flow_all'] = _rolling_sum(net_order_flow_all, W).astype(np.float32)
        progress.update(task, advance=1)

        mid_price_diff = np.diff(mid_price, prepend=mid_price[0]).astype(np.float32)
        ask_size_1_diff = np.diff(ask1_sz, prepend=ask1_sz[0]).astype(np.float32)
        bid_size_1_diff = np.diff(bid1_sz, prepend=bid1_sz[0]).astype(np.float32)

        buy_attack = np.maximum(np.float32(0.0), np.where(mid_price_diff > 0.0, -ask_size_1_diff, np.float32(0.0))).astype(np.float32)
        sell_attack = np.maximum(np.float32(0.0), np.where(mid_price_diff < 0.0, -bid_size_1_diff, np.float32(0.0))).astype(np.float32)

        features['cumulative_buy_attack'] = _rolling_sum(buy_attack, W).astype(np.float32)
        features['cumulative_sell_attack'] = _rolling_sum(sell_attack, W).astype(np.float32)
        total_attack = features['cumulative_buy_attack'] + features['cumulative_sell_attack']
        features['order_flow_imbalance'] = (
            (features['cumulative_buy_attack'] - features['cumulative_sell_attack']) / (total_attack + np.float32(1e-8))
        ).astype(np.float32)
        progress.update(task, advance=1)

        bid_near_depth = bid_size_arry[:, 0:4].sum(axis=1, dtype=np.float32)
        bid_far_depth  = bid_size_arry[:, 4:8].sum(axis=1, dtype=np.float32)
        bid_ratio = (bid_far_depth / (bid_near_depth + np.float32(1e-8))).astype(np.float32)
        ask_near_depth = ask_size_arry[:, 0:4].sum(axis=1, dtype=np.float32)
        ask_far_depth  = ask_size_arry[:, 4:8].sum(axis=1, dtype=np.float32)
        ask_ratio = (ask_far_depth / (ask_near_depth + np.float32(1e-8))).astype(np.float32)
        features['spoofing_proxy'] = (bid_ratio - ask_ratio).astype(np.float32)
        progress.update(task, advance=1)

    return pd.DataFrame(features, dtype=np.float32)

class SlidingDataset(Dataset):
    """
    Sliding window dataset for time series prediction.
    
    Args:
        data (pd.DataFrame or np.array): Input data matrix, last column should be target y.
        seq_len (int): Sequence length (Lookback window).
    """
    def __init__(self, data, seq_len):
        if isinstance(data, pd.DataFrame):
             self.data = torch.FloatTensor(data.values)
        else:
             self.data = torch.FloatTensor(data)
             
        self.seq_len = seq_len
        self.n_samples = len(data) - seq_len + 1
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Optimized slicing
        X = self.data[idx : idx + self.seq_len, :-1]
        y = self.data[idx + self.seq_len - 1, -1]
        return X, y

def construct_dataloader(df, seq_len, batch_size=512, num_workers=4, shuffle=False):
    """
    Construct PyTorch DataLoader.
    
    Args:
        df (pd.DataFrame): DataFrame containing features and target.
        seq_len (int): Sequence length.
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
        shuffle (bool): Whether to shuffle data.
        
    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = SlidingDataset(data=df, seq_len=seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# --- MODELS ---

class GRUModel(nn.Module):
    """
    GRU (Gated Recurrent Unit) based time series prediction model.
    
    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of GRU layers.
    """
    def __init__(self, input_dim=48, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        # Hidden state default is 0, so we don't strictly need to pass h0 unless we want stateful
        # But here we just want the output from the last time step
        _, h_n = self.gru(x)
        out = h_n[-1, :, :] 
        out = self.fc(out)
        return out

class OptimizedLassoRegression(nn.Module):
    """
    Optimized Lasso Regression with efficient GPU utilization.
    Includes single linear layer and Kaiming initialization.
    """
    def __init__(self, input_dim):
        super(OptimizedLassoRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        # Better initialization for faster convergence
        nn.init.kaiming_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def l1_regularization(model, alpha):
    """Compute L1 regularization (optimized for model parameters)"""
    l1_loss = sum(torch.abs(param).sum() for param in model.parameters())
    return alpha * l1_loss


# --- TRAINING LOGIC ---

def train_lasso_optimized(df_train, df_val, alpha, max_epochs, lr, batch_size, device, use_amp, optimizer_name):
    """
    Optimized PyTorch Lasso training process. Supports AMP and multiple optimizers.
    
    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        alpha (float): L1 regularization strength.
        max_epochs (int): Maximum training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (torch.device): Training device (CPU/GPU).
        use_amp (bool): Whether to use Automatic Mixed Precision (AMP).
        optimizer_name (str): Optimizer name ('adam', 'sgd', 'lbfgs').
        
    Returns:
        tuple: (best_model, best_epoch, best_val_loss) Best model, epoch, and validation loss.
    """
    """
    Highly optimized Lasso training with GPU acceleration and progress bars
    """
    console = Console()
    
    # Print header
    console.print("\n" + "="*70, style="bold cyan")
    console.print("OPTIMIZED PYTORCH LASSO TRAINING", style="bold yellow", justify="center")
    console.print("="*70 + "\n", style="bold cyan")
    
    # Prepare data (vectorized, no loops)
    console.print("📊 [bold]Preparing data...[/bold]")
    X_train, y_train = df_train.values[:, :-1], df_train.values[:, -1]
    X_val, y_val = df_val.values[:, :-1], df_val.values[:, -1]

    train_mask = np.isfinite(y_train)
    val_mask = np.isfinite(y_val)

    X_train, y_train = np.nan_to_num(X_train[train_mask]).astype(np.float32), y_train[train_mask].astype(np.float32)
    X_val, y_val = np.nan_to_num(X_val[val_mask]).astype(np.float32), y_val[val_mask].astype(np.float32)
    
    n_train, input_dim = X_train.shape
    n_val = len(X_val)
    
    # Create info table
    table = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=20)
    table.add_column("Value", style="green")
    table.add_row("Training Samples", f"{n_train:,}")
    table.add_row("Validation Samples", f"{n_val:,}")
    table.add_row("Input Dimension", str(input_dim))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Device", str(device))
    table.add_row("Mixed Precision", "✓ Enabled" if use_amp else "✗ Disabled")
    table.add_row("Optimizer", optimizer_name.upper())
    table.add_row("Learning Rate", f"{lr:.6f}")
    table.add_row("Alpha (L1)", f"{alpha:.6f}")
    console.print(table)
    console.print()
    
    # Convert to tensors and move to device ONCE
    console.print("🚀 [bold]Loading data to GPU...[/bold]")
    start_time = time.time()
    X_train_gpu = torch.from_numpy(X_train).to(device, non_blocking=True)
    y_train_gpu = torch.from_numpy(y_train).to(device, non_blocking=True)
    X_val_gpu = torch.from_numpy(X_val).to(device, non_blocking=True)
    y_val_gpu = torch.from_numpy(y_val).to(device, non_blocking=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
        console.print(f"✓ Data loaded to GPU in {time.time() - start_time:.2f}s", style="green")
        console.print(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved\n")
    
    # Initialize model
    model = OptimizedLassoRegression(input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    else:  # lbfgs
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Training state
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    console.print("="*70, style="bold cyan")
    console.print("🔥 [bold yellow]Starting Training[/bold yellow]")
    console.print("="*70 + "\n", style="bold cyan")
    
    # Training loop with progress bar
    epoch_pbar = tqdm(range(max_epochs), desc="Training Progress", unit="epoch", 
                      ncols=100, colour='green', position=0)
    
    training_start_time = time.time()
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Create random permutation for shuffling
        perm = torch.randperm(n_train, device=device)
        
        if optimizer_name == 'lbfgs':
            # LBFGS: full batch optimization
            def closure():
                optimizer.zero_grad()
                with autocast(enabled=use_amp):
                    predictions = model(X_train_gpu)
                    mse_loss = criterion(predictions, y_train_gpu)
                    l1_loss = l1_regularization(model, alpha)
                    loss = mse_loss + l1_loss
                
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                return loss
            
            if use_amp:
                scaler.step(optimizer, closure)
                scaler.update()
            else:
                optimizer.step(closure)
            
            with torch.no_grad():
                predictions = model(X_train_gpu)
                mse_loss = criterion(predictions, y_train_gpu)
                l1_loss = l1_regularization(model, alpha)
                epoch_loss = (mse_loss + l1_loss).item()
            
        else:
            # Adam/SGD: mini-batch optimization with progress bar
            n_total_batches = (n_train + batch_size - 1) // batch_size
            
            # Using simple range for batch iteration inside epoch to avoid nested tqdm clutter in this function context
            # or could use track()
            
            for i in range(0, n_train, batch_size):
                indices = perm[i:min(i + batch_size, n_train)]
                batch_X = X_train_gpu[indices]
                batch_y = y_train_gpu[indices]
                
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed precision forward pass
                with autocast(enabled=use_amp):
                    predictions = model(batch_X)
                    mse_loss = criterion(predictions, batch_y)
                    l1_loss = l1_regularization(model, alpha)
                    loss = mse_loss + l1_loss
                
                # Backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
            epoch_loss /= n_batches
        
        # Validation (efficient, no grad)
        model.eval()
        with torch.no_grad(), autocast(enabled=use_amp):
            val_predictions = model(X_val_gpu)
            val_mse = criterion(val_predictions, y_val_gpu)
            val_l1 = l1_regularization(model, alpha)
            val_loss = val_mse + val_l1
        
        val_loss_value = val_loss.item()
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss_value)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Calculate epoch time and GPU utilization
        epoch_time = time.time() - epoch_start_time
        
        if device.type == 'cuda':
            gpu_util = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        else:
            gpu_util = 0
        
        # Update progress bar description
        desc = f"Epoch {epoch+1}/{max_epochs} | Train: {epoch_loss:.6f} | Val: {val_loss_value:.6f}"
        epoch_pbar.set_description(desc)
        
        # Early stopping check
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                console.print(f"\n⏹️  [bold yellow]Early stopping at epoch {epoch+1}[/bold yellow] "
                            f"(no improvement for {patience} epochs)")
                break
    
    epoch_pbar.close()
    
    total_time = time.time() - training_start_time
    
    # Return best model
    model.load_state_dict(best_model_state)
    return model, best_epoch, best_val_loss


# --- PREDICTION LOGIC ---

def run_prediction_nn(test_data_path, model_path, scalar_path, output_path, seq_len=60, input_dim=48, hidden_dim=64, num_layers=2, batch_size=4096, device='cuda'):
    """
    Run NN prediction using pre-trained model.
    
    Args:
        test_data_path (str): Test data path (.csv.gz).
        model_path (str): Model weights path (.pth).
        scalar_path (str): Scaler path (.joblib).
        output_path (str): Prediction output path (.csv).
        seq_len, input_dim, hidden_dim, num_layers: Model architecture parameters.
        
    Returns:
        tuple: (MSE, IC) Mean Squared Error and Information Coefficient.
    """
    """
    Run NN prediction using pre-trained model
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    
    device = torch.device(device)
    print(f"Using device: {device}")

    # 1. Load Scaler
    if not os.path.exists(scalar_path):
        raise FileNotFoundError(f"Scaler not found at {scalar_path}")
    print(f"Loading scaler: {scalar_path}")
    scaler = load(scalar_path)

    # 2. Load Data
    print(f"Loading test data: {test_data_path}")
    df_head = pd.read_csv(test_data_path, nrows=1)
    if "Unnamed: 0" in df_head.columns:
         df = pd.read_csv(test_data_path, compression='gzip', index_col=0)
    else:
         df = pd.read_csv(test_data_path, compression='gzip')
    
    # 3. Preprocess
    # origin uses all columns except y as features
    # Assuming 'y' is the last column or named 'y'
    if 'y' in df.columns:
        X_raw = df.drop(columns=['y']).values
        y_raw = df['y'].values
    else:
        X_raw = df.values[:, :-1]
        y_raw = df.values[:, -1]
    
    print("Transforming data...")
    X_scaled = scaler.transform(X_raw)
    X_scaled = np.nan_to_num(X_scaled)
    
    data_matrix = np.column_stack((X_scaled, y_raw))
    
    # 4. Dataset
    print("Creating dataset...")
    dataset = SlidingDataset(data_matrix, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 5. Load Model
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
         state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
         state_dict = checkpoint['model_state_dict']
    else:
         state_dict = checkpoint
         
    model = GRUModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 6. Predict
    print("Predicting...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in track(dataloader, description="Inference"):
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 7. Metrics
    mse = np.mean((all_preds - all_targets)**2)
    ic = np.corrcoef(all_preds, all_targets)[0, 1]
    
    print("="*50)
    print(f"NN Test Results (OOS)")
    print(f"MSE: {mse:.6f}")
    print(f"IC: {ic:.6f}")
    print("="*50)
    
    # 8. Save
    pd.DataFrame({'y_true': all_targets, 'y_pred': all_preds}).to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return mse, ic


# --- ANALYSIS & VISUALIZATION ---

def analyze_feature_correlations_final(csv_path, sample_size=300000):
    """
    Calculate Pearson correlation between features and target y.
    Automatically identifies Rate, Size, Nc, etc. 8-level features.
    
    Args:
        csv_path (str): Data file path.
        sample_size (int): Number of samples to read for estimation.
        
    Returns:
        pd.DataFrame: DataFrame containing feature names and correlation coefficients.
    """
    console.print(f"[cyan]Reading data (final): {csv_path}")
    
    df = pd.read_csv(csv_path, nrows=sample_size)
    
    if 'y' not in df.columns:
        console.print("[red]❌ Critical Error: 'y' column not found in data! Please check file.")
        return None
        
    console.print(f"[green]✅ Found target column 'y'. Correlation calculation will proceed directly.")
    
    # Define feature groups
    levels = range(8)
    feature_types = ['askRate', 'bidRate', 'askSize', 'bidSize', 'askNc', 'bidNc']
    
    feature_cols = []
    for ftype in feature_types:
        for i in levels:
            col_name = f"{ftype}_{i}"
            if col_name in df.columns:
                feature_cols.append(col_name)
    
    console.print(f"[yellow]Identified {len(feature_cols)} valid feature columns (Rate/Size/Nc * 8 levels)")
    
    correlations = []
    target = df['y'].values
    
    for f in feature_cols:
        feat_val = df[f].values
        mask = np.isfinite(feat_val) & np.isfinite(target)
        if mask.sum() > 100:
            c = np.corrcoef(feat_val[mask], target[mask])[0, 1]
            if not np.isnan(c):
                correlations.append({'Feature': f, 'Correlation': c})
    
    res_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=True)
    return res_df

def plot_tornado(df, output_path):
    """
    Plot feature importance Tornado Chart.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Feature' and 'Correlation' columns.
        output_path (str): Image save path.
    """
    df = df.sort_values('Correlation')
    
    top_neg = df.head(15)
    top_pos = df.tail(15)
    plot_df = pd.concat([top_neg, top_pos])
    
    features = plot_df['Feature'].tolist()
    formatted_features = []
    for f in features:
        parts = f.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            name = parts[0]
            idx = int(parts[1])
            formatted_features.append(f"{name}_{idx + 1}")
        else:
            formatted_features.append(f)
            
    features = formatted_features
    values = plot_df['Correlation'].tolist()
    
    colors = ['#2b8cbe' if x < 0 else '#e41a1c' for x in values]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black')
    
    max_abs_val = max(abs(min(values)), abs(max(values)))
    limit_padding = max_abs_val * 0.15 
    ax.set_xlim(-max_abs_val - limit_padding, max_abs_val + limit_padding)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=16)
    ax.axvline(0, color='black', linewidth=0.8)
    
    for i, v in enumerate(values):
        offset = 0.0005 if v >= 0 else -0.0005
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + offset, i, f'{v:.4f}', va='center', ha=ha, fontsize=12)
    
    ax.set_title('Market Compass: Feature Correlations with Target (y)', fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('Pearson Correlation with Target y', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    console.print(f"[green]✅ Tornado chart generated: {output_path}")

def plot_performance(y_true, y_pred, title, output_path):
    """
    Plot prediction performance (Hexbin Plot + Regression Line).
    
    Args:
        y_true (np.array): True values.
        y_pred (np.array): Predicted values.
        title (str): Chart title.
        output_path (str): Image save path.
    """
    plt.figure(figsize=(16, 14))
    hb = plt.hexbin(y_pred, y_true, gridsize=50, cmap='inferno', mincnt=1, bins='log')
    cb = plt.colorbar(hb)
    cb.set_label('log10(Count)', fontsize=28)
    cb.ax.tick_params(labelsize=24)

    m, b = np.polyfit(y_pred, y_true, 1)
    ic = np.corrcoef(y_true, y_pred)[0,1]
    plt.plot(y_pred, m*y_pred + b, color='cyan', linewidth=3, linestyle='--', label=f'Fit Line (IC={ic:.4f})')

    plt.xlim(y_pred.min(), y_pred.max())
    plt.ylim(y_true.min(), y_true.max())

    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('Predicted Return', fontsize=28)
    plt.ylabel('Actual Return (y)', fontsize=28)
    plt.title(f'{title} Performance: R^2={r2:.6f}', fontsize=30)
    plt.legend(fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to {output_path}")
    plt.close()

def optimize_weights(predictions, y_true):
    """
    Optimize ensemble model weights using SLSQP to minimize MSE.
    
    Args:
        predictions (np.array): Prediction matrix (Samples, N_Models).
        y_true (np.array): True values array.
        
    Returns:
        np.array: Optimal weights array (sum=1, w>=0).
    """
    n_models = predictions.shape[1]
    initial_weights = np.ones(n_models) / n_models
    
    def mse_loss(weights):
        combined_pred = np.dot(predictions, weights)
        return np.mean((y_true - combined_pred) ** 2)
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(n_models))
    
    result = minimize(mse_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def load_prediction(path, model_name):
    """Load prediction file (parquet or csv) and return dataframe with y and score."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file for {model_name} not found: {path}")
    
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    if 'y' not in df.columns or 'score' not in df.columns:
        if 'y_true' in df.columns: df = df.rename(columns={'y_true': 'y'})
        if 'y_pred' in df.columns: df = df.rename(columns={'y_pred': 'score'})
        if 'prediction' in df.columns: df = df.rename(columns={'prediction': 'score'})
    
    print(f"Loaded {model_name}: {len(df)} samples")
    return df[['y', 'score']].rename(columns={'score': f'score_{model_name}'})


# --- ADDITIONAL MODELS & TRAINING ---

class MixerBlock(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len),
            nn.GELU()
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = x + self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(x)
        return x
    
class MLPMixer(nn.Module):
    def __init__(self, input_dim=21, seq_len=60, hidden_dim=64, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MixerBlock(seq_len, hidden_dim))
    
        self.output_layer = nn.Linear(hidden_dim * seq_len , 1)
        
    def forward(self, x):
        x = self.input_proj(x)  # (b, 60, 64)
        for layer in self.layers:
            x = layer(x)  # (b, 60, 64)
        x = x.reshape(x.shape[0], -1)
        x = self.output_layer(x)  # (b, 1)
        return x

def train_nn_model(
    model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int = 100,
    patience: int = 5,
    device: str = "cuda"
):
    """
    Trains a PyTorch model (Generic) on a single GPU or CPU.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    console = Console()
    progress = Progress(console=console)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0
    train_losses, val_losses = [], []
    
    with progress:
        for epoch in range(num_epochs):
            # --- Training Phase ---
            model.train()
            train_loss = 0
            train_task = progress.add_task(f"[green]Epoch {epoch+1} Training...", total=len(train_dataloader))
            
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                progress.update(train_task, advance=1, description=f"[green]Epoch {epoch+1} Training... [Loss: {loss.item():.4f}]")
            
            avg_train_loss = train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)

            # --- Validation Phase ---
            model.eval()
            val_loss = 0
            val_task = progress.add_task(f"[blue]Epoch {epoch+1} Validation...", total=len(val_dataloader))
            
            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    loss = criterion(outputs.squeeze(), y)
                    val_loss += loss.item()
                    progress.update(val_task, advance=1, description=f"[blue]Epoch {epoch+1} Validation... [Loss: {loss.item():.4f}]")
            
            avg_val_loss = val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

            progress.print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, epochs_no_improve: {epochs_no_improve}/{patience}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    console.print(f"[bold red]Early stopping triggered at epoch {epoch+1}!")
                    break
    
    model.load_state_dict(best_model_state)
    return model, best_epoch, best_val_loss

def train_xgb_pytorch(df_train, df_val, params, save_path=None):
    """
    Train XGBoost model with PyTorch-compatible data handling
    """
    if xgb is None:
        raise ImportError("xgboost is not installed.")

    # Prepare data
    X_train, y_train = df_train.values[:, :-1], df_train.values[:, -1]
    X_val, y_val = df_val.values[:, :-1], df_val.values[:, -1]

    train_mask, val_mask = np.isfinite(y_train), np.isfinite(y_val)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=True
    )
    
    if save_path:
        model.save_model(save_path)
        print(f"XGBoost model saved to {save_path}")
    
    val_pred = model.predict(X_val)
    val_mse = np.mean((val_pred - y_val) ** 2)
    val_corr = np.corrcoef(val_pred, y_val)[0, 1]
    
    print(f"\nValidation Results:")
    print(f"MSE: {val_mse:.6f}")
    print(f"Correlation: {val_corr:.6f}")
    
    return model, val_mse, val_corr

class SimpleLassoRegression(nn.Module):
    """
    Basic PyTorch implementation of Lasso Regression
    """
    def __init__(self, input_dim):
        super(SimpleLassoRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x).squeeze()

def train_lasso_simple(df_train, df_val, alpha, epochs, lr, batch_size, device):
    """
    Train Lasso regression using PyTorch (Simple/Basic version)
    """
    # Prepare data
    X_train, y_train = df_train.values[:, :-1], df_train.values[:, -1]
    X_val, y_val = df_val.values[:, :-1], df_val.values[:, -1]

    train_mask, val_mask = np.isfinite(y_train), np.isfinite(y_val)

    X_train, y_train = np.nan_to_num(X_train[train_mask]), y_train[train_mask]
    X_val, y_val = np.nan_to_num(X_val[val_mask]), y_val[val_mask]
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    model = SimpleLassoRegression(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    print(f"Starting Simple Lasso training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            mse_loss = criterion(predictions, batch_y)
            l1_loss = l1_regularization(model, alpha)
            loss = mse_loss + l1_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_mse = criterion(val_predictions, y_val_tensor)
            val_l1 = l1_regularization(model, alpha)
            val_loss = val_mse + val_l1
        
        avg_train_loss = train_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    return model, best_val_loss.item()


def model_predict_xgb(model_path, df_test, output_dir=None):
    """
    Run prediction using XGBoost.
    """
    if xgb is None:
        raise ImportError("xgboost is not installed.")

    # Load the model from the specified path
    model = joblib.load(model_path)
    
    # Get the target variable 'y' from the test DataFrame
    # Handle case where y might not be the last column or might be named 'y'
    if 'y' in df_test.columns:
        X_test = df_test.drop(columns=['y']).values
        y_test = df_test['y'].values
    else:
        X_test, y_test = df_test.values[:, :-1], df_test.values[:, -1]
        
    test_mask = np.isfinite(y_test)
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    # Predict using the model
    y_pred = model.predict(X_test)
    
    # Create a DataFrame with the original 'y' values and predicted 'score'
    result_df = pd.DataFrame({
        'y': y_test,           # The true 'y' values
        'score': y_pred   # The predicted values (score)
    })
    
    # Try to align index if possible
    try:
        if 'y' in df_test.columns:
             filtered_index = df_test.index[test_mask]
        else:
             filtered_index = df_test.index[test_mask] # assumes standard range index or similar
        result_df.index = filtered_index
    except Exception:
        pass # ignore index alignment if it fails
        
    r2 = r2_score(result_df['y'], result_df['score'])
    ic = np.corrcoef(result_df['y'], result_df['score'])[0][1]
    
    print("R-square:", r2)
    print("IC:", ic)
    
    # Save the predictions if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pred_file = os.path.join(output_dir, 'xgb_result.parquet')
        result_df.to_parquet(pred_file)
        print(f"Predictions saved to {pred_file}")

    return result_df

def construct_predictions(df, valid_slices, pred, seq_len):
    """
    Constructs a results DataFrame using the exact indices from the dataset.
    """
    # map valid_slices (start indices) to target indices (end of sequence)
    target_indices = [t + seq_len - 1 for t in valid_slices]
    
    # Retrieve actual values and timestamps using the indices
    y_true = df.iloc[target_indices].iloc[:, -1].values
    timestamps = df.index[target_indices]
    
    df_res = pd.DataFrame({
        'y': y_true,
        'score': pred
    }, index=timestamps)
    
    ic = np.corrcoef(df_res['y'], df_res['score'])[0, 1]
    r_square = r2_score(df_res['y'], df_res['score'])
    print(f"IC on dataset: {ic}")
    print(f"R-square on dataset: {r_square}")
    return df_res

def model_pred_nn(test_df, model_dir, seq_len=60, input_dim=48, hidden_dim=64, layer_nums=2, model_name="GRUModel", output_dir=None):
    """
    Run prediction using NN model (Generic).
    """
    # Create dataloader which initializes the dataset and computes valid_slices
    test_dataloader = construct_dataloader(test_df, batch_size=2048, seq_len=seq_len, shuffle=False)
    
    # Access the dataset to get the valid indices
    dataset = test_dataloader.dataset
    valid_slices = dataset.valid_slices
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model_dir is a file or directory
    if os.path.isdir(model_dir):
        # find .pth file
        pth_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if pth_files:
             pth_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
             model_path = os.path.join(model_dir, pth_files[0])
        else:
             raise FileNotFoundError(f"No .pth files found in {model_dir}")
    else:
        model_path = model_dir

    state_dict = torch.load(model_path, map_location=device)
    
    # Handle nested state_dict if necessary
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    if model_name == "MLPMixer":
        model = MLPMixer(
            input_dim=input_dim,  
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            num_layers=layer_nums
        )
    elif model_name == "GRUModel":
        model = GRUModel(
            input_dim=input_dim,  
            hidden_dim=hidden_dim,
            num_layers=layer_nums
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        # quick fix try removing 'module.' prefix if it exists (from DataParallel)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for X, y in track(test_dataloader, description="Predicting..."):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predictions.extend(outputs.cpu().numpy().flatten().tolist())
                
    # Ensure predictions match the number of valid slices
    if len(predictions) != len(valid_slices):
        print(f"[Warning] Mismatch: len(predictions)={len(predictions)}, len(valid_slices)={len(valid_slices)}")
        min_len = min(len(predictions), len(valid_slices))
        predictions = predictions[:min_len]
        valid_slices = valid_slices[:min_len]

    res_df = construct_predictions(test_df, valid_slices, predictions, seq_len) 
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        res_df.to_parquet(os.path.join(output_dir, 'result_df.parquet'))
        print(f"Predictions saved to {os.path.join(output_dir, 'result_df.parquet')}")
        
    return res_df


class SimpleLassoRegression(nn.Module): # Redefining here if needed but it's already defined above. 
    # Actually, we should check if the previous definition covers it. 
    # The previous `SimpleLassoRegression` matches `LassoRegression` in predict_lasso_pytorch.py 
    # except for initialization details which don't matter for inference.
    pass

def model_predict_lasso_pytorch(df_test, model_file, device="cuda"):
    """
    Make predictions using PyTorch Lasso model
    """
    # Load model checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    if 'input_dim' in checkpoint:
        input_dim = checkpoint['input_dim']
    else:
        # Fallback if input_dim not saved, infer from data
        input_dim = df_test.shape[1] - 1
        print(f"Warning: input_dim not found in checkpoint, using {input_dim} from data")

    # Initialize model with correct dimensions
    # Using OptimizedLassoRegression logic cause it's compatible for forward pass (linear layer)
    # Or SimpleLassoRegression
    model = SimpleLassoRegression(input_dim).to(device) 
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # In case it's just the state dict
        
    model.eval()
    
    print(f"Loaded model from {model_file}")
    
    # Prepare test data
    X_test = df_test.values[:, :-1]
    y_test = df_test.values[:, -1]
    
    test_mask = np.isfinite(y_test)
    X_test_valid = np.nan_to_num(X_test[test_mask])
    y_test_valid = y_test[test_mask]
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test_valid).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
    
    # Calculate metrics
    mse = np.mean((predictions - y_test_valid) ** 2)
    correlation = np.corrcoef(predictions, y_test_valid)[0, 1]
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.6f}")
    print(f"Correlation: {correlation:.6f}")
    
    # Create result dataframe
    result_df = pd.DataFrame({
        'y': y_test_valid,
        'score': predictions
    })
    
    return result_df, mse, correlation

def ensemble_predictions(result_paths, weights=None):
    """
    Ensemble predictions from multiple parquet files.
    
    Args:
        result_paths (list of str): List of paths to result_df.parquet files.
        weights (list of float, optional): Weights for each model. If None, average.
    """
    dfs = []
    for path in result_paths:
        if not os.path.exists(path):
             print(f"Warning: {path} not found. Skipping.")
             continue
        dfs.append(pd.read_parquet(path))
    
    if not dfs:
        raise ValueError("No valid result files found.")

    if weights is None:
        weights = np.ones(len(dfs)) / len(dfs)
    else:
        weights = np.array(weights)
        
    y_col = dfs[0]['y'].copy()
    
    # Ensure all dfs have same index/length
    # This assumes they are aligned.
    
    scores = np.zeros_like(y_col)
    
    for i, df in enumerate(dfs):
        scores += df['score'].values * weights[i]
        
    final_df = pd.DataFrame({'y': y_col, 'score': scores}, index=dfs[0].index)
    
    r2 = r2_score(final_df['y'], final_df['score'])
    ic = np.corrcoef(final_df['y'], final_df['score'])[0, 1]
    
    print(f"Ensemble R2: {r2}")
    print(f"Ensemble IC: {ic}")
    
    return final_df


# --- ADVANCED MODELS & UTILS ---

class GRUModelMagic(nn.Module):
    """
    GRU Model with specific feature selection indices.
    """
    def __init__(self, input_dim=21, hidden_dim=128, num_layers=2):
        super().__init__()
        self.mixer_indices = [1,2,7,10,12,13,15,16,18,19,20]
        self.linear_indices = [2,7,10,13,16]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=len(self.mixer_indices),
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear_layer = nn.Linear(len(self.linear_indices), 1)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x_mixer = x[:, :, self.mixer_indices]
        x_linear = x[:, -1, self.linear_indices]
        # h0 defaults to zeros if not provided
        out, _ = self.gru(x_mixer)  
        out = out[:, -1, :]  
        z1 = self.fc(out) 
        z2 = self.linear_layer(x_linear)
        return z1 + z2

class MLPMixerMagic(nn.Module):
    """
    MLP Mixer with specific feature selection and learnable combination weight.
    """
    def __init__(self, input_dim=21, seq_len=60, hidden_dim=64, num_layers=4):
        super().__init__()
        self.linear_indices = [2,7,10,13,16]
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MixerBlock(seq_len, hidden_dim))
            
        self.linear_input_dim = len(self.linear_indices)
        self.linear_layer = nn.Linear(self.linear_input_dim, 1)
    
        self.output_layer = nn.Linear(hidden_dim * seq_len, 1)
        
        self.comb_weight = nn.Parameter(torch.tensor(0.5)) 
        
    def forward(self, x):
        x_linear = x[:, -1, self.linear_indices]
        x = self.input_proj(x) 
        
        for layer in self.layers:
            x = layer(x)  
        x = x.reshape(x.shape[0], -1)
        z1 = self.output_layer(x)
        z2 = self.linear_layer(x_linear)
        w = torch.sigmoid(self.comb_weight) 
        return (1 - w) * z1 + w * z2

def get_robust_scaler(train_path, scaler_path, source_type="origin"):
    """
    Get or fit a RobustScaler.
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    if os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}")
        return load(scaler_path)
    
    print(f"Fitting new scaler for {source_type}...")
    # origin source = raw 48 columns
    if source_type == "origin":
        df = pd.read_csv(train_path, nrows=100000, compression='gzip')
        X = df.iloc[:, :48].values
    else:
        # Fallback for other types, assuming standard csv structure
        df = pd.read_csv(train_path, nrows=100000, compression='gzip')
        if 'y' in df.columns:
             X = df.drop(columns=['y']).values
        else:
             X = df.values[:, :-1]
    
    scaler = RobustScaler()
    scaler.fit(X)
    dump(scaler, scaler_path)
    return scaler



