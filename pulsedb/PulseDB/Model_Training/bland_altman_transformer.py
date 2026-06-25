import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("/project/zouridakis/gzlab2/PulseDB/pulsedb_yidan_multi/Model_Training/Model_Def")
from mat73 import loadmat

def bland_altman_plot(y_true, y_pred, title="Bland-Altman Plot", save_path=None):
    """
    Create a Bland-Altman plot for assessing agreement between methods
    
    Parameters:
    y_true: array-like, ground truth values
    y_pred: array-like, predicted values
    title: str, plot title
    save_path: str, path to save the plot (optional)
    """
    # Convert to numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate differences and means
    differences = y_pred - y_true
    means = (y_true + y_pred) / 2
    
    # Calculate statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # Calculate 95% limits of agreement
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(means, differences, alpha=0.6, s=20, color='steelblue')
    
    # Add horizontal lines
    plt.axhline(mean_diff, color='red', linestyle='-', linewidth=1.5)
    plt.axhline(upper_limit, color='red', linestyle='--', linewidth=1.5)
    plt.axhline(lower_limit, color='red', linestyle='--', linewidth=1.5)
    
    # Add text annotations
    plt.text(plt.xlim()[1] * 0.95, upper_limit, f'+1.96 SD\n{upper_limit:.2f}', 
             ha='right', va='bottom', fontsize=10, color='red')
    plt.text(plt.xlim()[1] * 0.95, mean_diff, f'Mean\n{mean_diff:.2f}', 
             ha='right', va='bottom', fontsize=10, color='red')
    plt.text(plt.xlim()[1] * 0.95, lower_limit, f'-1.96 SD\n{lower_limit:.2f}', 
             ha='right', va='top', fontsize=10, color='red')
    
    # Set labels and title
    plt.xlabel('Mean of methods', fontsize=12)
    plt.ylabel('Percentage difference between methods', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print(f"Mean difference: {mean_diff:.2f}")
    print(f"Standard deviation: {std_diff:.2f}")
    print(f"95% Limits of Agreement: [{lower_limit:.2f}, {upper_limit:.2f}]")
    
    return mean_diff, std_diff, lower_limit, upper_limit

def predict_from_data(model, signals, static_features, device, batch_size=128):
    """
    Generate predictions from signal data using a trained model
    
    Parameters:
    model: trained PyTorch model
    signals: numpy array of shape (n_samples, n_channels, signal_length)
    device: torch device
    batch_size: batch size for inference
    
    Returns:
    predictions: numpy array of predictions
    """
    
    predictions = []
    
    # Convert to tensors
    signals_tensor = torch.from_numpy(signals).float()
    static_tensor = torch.from_numpy(static_features).float()
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, len(signals_tensor), batch_size):
            signal_batch = signals_tensor[i:i+batch_size].to(device)
            static_batch = static_tensor[i:i+batch_size].to(device)
            outputs = model(signal_batch, static_batch)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0).flatten()

# Example usage with your approach:
def run_bland_altman_analysis(data_path, model_path, bp_type='SBP', signal_channel=1):
    """
    Complete Bland-Altman analysis from data file and model
    
    Parameters:
    data_path: path to .mat file
    model_path: path to trained model
    bp_type: 'SBP' or 'DBP'
    signal_channel: which signal channel to use (1 for PPG)
    """
    # Load data
    data = loadmat(data_path)
    
    # Extract signals and labels
    signals = data['Subset']['Signals'][:, signal_channel:signal_channel+1, :]  # PPG channel
    labels = data['Subset'][bp_type].flatten()
    
    # Extract static features (Age, Gender, Height)
    age = data['Subset']['Age'].flatten()
    gender = np.array([1.0 if g == 'M' else 0.0 for g in data['Subset']['Gender']])  # M=1, F=0
    height = data['Subset']['Height'].flatten()
    
    # Create static features array - 3 features as expected by model
    static_features = np.column_stack([age, gender, height])

    # Setup model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Generate predictions
    predictions = predict_from_data(model, signals, static_features, device)
    
    # Create Bland-Altman plot
    title = f"Bland-Altman plot for {bp_type}"
    stats = bland_altman_plot(labels, predictions, title=title, save_path="bland_altman_plot_transformer.png")
    
    return labels, predictions, stats

# Your usage:

import Mtransformer  # or mtransformer, depending on actual filename
import types

# Create expected module structure
Model_Def = types.ModuleType('Model_Def')
Model_Def.Mtransformer = Mtransformer
sys.modules['Model_Def'] = Model_Def
sys.modules['Model_Def.Mtransformer'] = Mtransformer

data_path = '/project/zouridakis/gzlab2/PulseDB/Subset_Files/CalFree_Test_Subset_filtered.mat'
model_path = '/project/zouridakis/gzlab2/142412/checkpoint_epoch_75.pth'

# Run analysis
y_true, y_pred, stats = run_bland_altman_analysis(data_path, model_path, 'SBP')
