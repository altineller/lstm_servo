# train.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# ========== DATA PROCESSOR ==========
class RPMDataProcessor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = [
            'target_left', 'target_right',
            'cs_left', 'cs_right', 
            'pwm_left', 'pwm_right'
        ]
        self.target_columns = ['rpm_left', 'rpm_right']
    
    def prepare_features(self, df):
        """Prepare input features and target variables"""
        # Ensure we have all required columns
        missing_features = [col for col in self.feature_columns + self.target_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing columns in dataframe: {missing_features}")
        
        # Create sequences
        X, y = self.create_sequences(df[self.feature_columns].values, 
                                   df[self.target_columns].values)
        
        return X, y
    
    def create_sequences(self, data, targets):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(targets[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def fit_scalers(self, X, y):
        """Fit scalers to training data"""
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler_x.fit(X_reshaped)
        self.scaler_y.fit(y)
    
    def transform_x(self, X):
        """Transform input data using fitted scaler"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler_x.transform(X_reshaped)
        return X_scaled.reshape(original_shape)
    
    def transform_y(self, y):
        """Transform target data using fitted scaler"""
        return self.scaler_y.transform(y)
    
    def inverse_transform_y(self, y_scaled):
        """Inverse transform target data"""
        return self.scaler_y.inverse_transform(y_scaled)

# ========== LSTM MODEL ==========
class RPMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2, dropout=0.2):
        super(RPMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Additional layers for better learning
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.dropout(last_output)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        
        return x

# ========== TRAINER ==========
class RPMPredictionTrainer:
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=128, early_stopping_patience=20):
        """Complete training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.train_losses, self.val_losses
    
    def predict(self, dataloader):
        """Make predictions"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                batch_predictions = self.model(X_batch)
                predictions.append(batch_predictions.cpu().numpy())
                actuals.append(y_batch.cpu().numpy())
        
        return np.vstack(predictions), np.vstack(actuals)

# ========== DATA GENERATOR ==========
def generate_sample_data(n_samples=5000):
    """Generate realistic sample data for testing"""
    np.random.seed(42)
    
    time = np.linspace(0, 128, n_samples)
    
    # Simulate motor dynamics with some latency and inertia
    def motor_dynamics(target, delay=5, inertia=0.1):
        rpm = np.zeros_like(target)
        for i in range(len(target)):
            if i < delay:
                rpm[i] = 0
            else:
                # Simple first-order system with some noise
                rpm[i] = (inertia * rpm[i-1] + (1-inertia) * target[i-delay] + 
                         np.random.normal(0, 0.5))
        return rpm
    
    # Generate realistic target commands
    target_left = np.zeros(n_samples)
    target_right = np.zeros(n_samples)
    
    # Create various command patterns
    for i in range(0, n_samples, 500):
        if i + 500 < n_samples:
            pattern = np.random.choice(['step', 'ramp', 'sine', 'zero'])
            if pattern == 'step':
                target_left[i:i+500] = np.random.uniform(-50, 50)
                target_right[i:i+500] = np.random.uniform(-50, 50)
            elif pattern == 'ramp':
                target_left[i:i+500] = np.linspace(-30, 30, 500)
                target_right[i:i+500] = np.linspace(-25, 35, 500)
            elif pattern == 'sine':
                freq = np.random.uniform(0.1, 0.5)
                target_left[i:i+500] = 40 * np.sin(2 * np.pi * freq * np.arange(500) / 500)
                target_right[i:i+500] = 35 * np.sin(2 * np.pi * (freq + 0.1) * np.arange(500) / 500)
            else:  # zero
                target_left[i:i+500] = 0
                target_right[i:i+500] = 0
    
    # Generate RPM with motor dynamics
    rpm_left = motor_dynamics(target_left, delay=3, inertia=0.2)
    rpm_right = motor_dynamics(target_right, delay=4, inertia=0.15)
    
    # Generate PWM (controller output)
    pwm_left = np.clip(target_left * 5 + np.random.normal(0, 10, n_samples), -255, 255)
    pwm_right = np.clip(target_right * 5 + np.random.normal(0, 10, n_samples), -255, 255)
    
    # Generate current sense (torque feedback)
    cs_left = np.abs(rpm_left) * 0.001 + np.abs(pwm_left) * 0.0001 + np.random.normal(0, 0.001, n_samples)
    cs_right = np.abs(rpm_right) * 0.001 + np.abs(pwm_right) * 0.0001 + np.random.normal(0, 0.001, n_samples)
    
    df = pd.DataFrame({
        'timestamp': time,
        'target_left': target_left,
        'target_right': target_right,
        'rpm_left': rpm_left,
        'rpm_right': rpm_right,
        'pwm_left': pwm_left,
        'pwm_right': pwm_right,
        'cs_left': cs_left,
        'cs_right': cs_right
    })
    
    return df

# ========== MAIN FUNCTION ==========
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or generate data
    try:
        # Try to load your actual data
        df = pd.read_csv('data.csv')
        print("Loaded data from 'data.csv'")
    except:
        # Generate sample data if real data not available
        print("No data file found, generating sample data...")
        df = generate_sample_data(n_samples=5000)
        df.to_csv('sample_data.csv', index=False)
        print("Sample data saved to 'sample_data.csv'")
    
    print(f"Loaded data with {len(df)} samples")
    print("Data columns:", df.columns.tolist())
    
    # Data preparation
    processor = RPMDataProcessor(sequence_length=20)
    X, y = processor.prepare_features(df)
    
    print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    
    # Split data (maintaining temporal order)
    split_idx = int(0.7 * len(X))
    X_train, X_temp = X[:split_idx], X[split_idx:]
    y_train, y_temp = y[:split_idx], y[split_idx:]
    
    val_idx = int(0.5 * len(X_temp))
    X_val, X_test = X_temp[:val_idx], X_temp[val_idx:]
    y_val, y_test = y_temp[:val_idx], y_temp[val_idx:]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Scale data
    processor.fit_scalers(X_train, y_train)
    X_train_scaled = processor.transform_x(X_train)
    y_train_scaled = processor.transform_y(y_train)
    X_val_scaled = processor.transform_x(X_val)
    y_val_scaled = processor.transform_y(y_val)
    X_test_scaled = processor.transform_x(X_test)
    y_test_scaled = processor.transform_y(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.FloatTensor(y_train_scaled)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled), 
        torch.FloatTensor(y_val_scaled)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled), 
        torch.FloatTensor(y_test_scaled)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = RPMPredictor(
        input_size=input_size, 
        hidden_size=64, 
        num_layers=2, 
        output_size=2, 
        dropout=0.3
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    trainer = RPMPredictionTrainer(model, learning_rate=0.001, device=device)
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=200)
    
    # Save trained model
    torch.save(model.state_dict(), 'rpm_predictor_model.pth')
    print("Model saved to 'rpm_predictor_model.pth'")
    
    # Evaluate model
    predictions_scaled, actuals_scaled = trainer.predict(test_loader)
    
    # Inverse transform predictions
    predictions = processor.inverse_transform_y(predictions_scaled)
    actuals = processor.inverse_transform_y(actuals_scaled)
    
    # Calculate metrics
    rmse_left = np.sqrt(np.mean((predictions[:, 0] - actuals[:, 0]) ** 2))
    rmse_right = np.sqrt(np.mean((predictions[:, 1] - actuals[:, 1]) ** 2))
    
    mae_left = np.mean(np.abs(predictions[:, 0] - actuals[:, 0]))
    mae_right = np.mean(np.abs(predictions[:, 1] - actuals[:, 1]))
    
    print(f"\n=== Test Results ===")
    print(f"Left RPM - RMSE: {rmse_left:.4f}, MAE: {mae_left:.4f}")
    print(f"Right RPM - RMSE: {rmse_right:.4f}, MAE: {mae_right:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot training history
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.yscale('log')
    
    # Plot left RPM predictions
    plt.subplot(2, 2, 2)
    n_plot = min(200, len(actuals))
    plt.plot(actuals[:n_plot, 0], label='Actual Left RPM', alpha=0.7, linewidth=2)
    plt.plot(predictions[:n_plot, 0], label='Predicted Left RPM', alpha=0.7, linestyle='--')
    plt.legend()
    plt.title('Left RPM Prediction')
    plt.ylabel('RPM')
    
    # Plot right RPM predictions
    plt.subplot(2, 2, 3)
    plt.plot(actuals[:n_plot, 1], label='Actual Right RPM', alpha=0.7, linewidth=2)
    plt.plot(predictions[:n_plot, 1], label='Predicted Right RPM', alpha=0.7, linestyle='--')
    plt.legend()
    plt.title('Right RPM Prediction')
    plt.ylabel('RPM')
    plt.xlabel('Time Step')
    
    # Plot scatter plots
    plt.subplot(2, 2, 4)
    plt.scatter(actuals[:, 0], predictions[:, 0], alpha=0.5, label='Left RPM', s=10)
    plt.scatter(actuals[:, 1], predictions[:, 1], alpha=0.5, label='Right RPM', s=10)
    max_val = max(np.max(actuals), np.max(predictions))
    min_val = min(np.min(actuals), np.min(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('Actual RPM')
    plt.ylabel('Predicted RPM')
    plt.legend()
    plt.title('Prediction vs Actual')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
