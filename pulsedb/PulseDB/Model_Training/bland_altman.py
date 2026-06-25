import numpy as np
import matplotlib.pyplot as plt
import torch
from mat73 import loadmat

# Add imports for model classes
from Model_Def.MResNet18 import MResnet18_1D
from Model_Def.MResNet50 import MResnet50_1D
from Model_Def.MS4 import MS4_1D
from Model_Def.Minception import MInception1d
from Model_Def.MLenet import MLeNet1d

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

def predict_from_data(model, signals, static_feat, device, batch_size=128):
    """
    Generate predictions from signal data using a trained model

    Parameters:
    model: trained PyTorch model
    signals: numpy array of shape (n_samples, n_channels, signal_length)
    static_feat: numpy array of shape (n_samples, n_static_features)
    device: torch device
    batch_size: batch size for inference

    Returns:
    predictions: numpy array of predictions
    """

    predictions = []

    # Convert to tensors
    signals_tensor = torch.from_numpy(signals).float()
    static_tensor = torch.from_numpy(static_feat).float()

    # Process in batches
    with torch.no_grad():
        for i in range(0, len(signals_tensor), batch_size):
            batch_signals = signals_tensor[i:i+batch_size].to(device)
            batch_static = static_tensor[i:i+batch_size].to(device)
            outputs = model(batch_signals, batch_static)
            predictions.append(outputs.cpu().numpy())

    return np.concatenate(predictions, axis=0).flatten()

def load_model(model_path, model_class, device):
    """Load a single model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        if model_class.__name__ == 'MS4_1D':
            print('No MS4 MODEL')
        else:
            # Try different common keys where the model might be stored
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                model = model_class()
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model = model_class()
                model.load_state_dict(checkpoint['state_dict'])
            elif 'net' in checkpoint:
                model = checkpoint['net']
            else:
                model = model_class()
                model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    
    model.to(device)
    model.eval()
    return model

# Example usage with your approach:
def run_bland_altman_analysis(data_path, model_path, model_class, bp_type='SBP', signal_channel=1):
    """
    Complete Bland-Altman analysis from data file and model

    Parameters:
    data_path: path to .mat file
    model_path: path to trained model
    model_class: model class for loading
    bp_type: 'SBP' or 'DBP'
    signal_channel: which signal channel to use (1 for PPG)
    """
    # Load data
    data = loadmat(data_path)

    # Extract signals, static features, and labels
    signals = data['Subset']['Signals'][:, signal_channel:signal_channel+1, :]  # PPG channel
    age = data['Subset']['Age']
    bmi = data['Subset']['BMI']
    gender = data['Subset']['Gender']
    gender_num = np.array([1.0 if g[0] == 'M' else 0.0 for g in gender])  # Convert to numeric
    static_feat = np.column_stack([age, gender_num, bmi])  # Shape: (n_samples, 3)
    labels = data['Subset'][bp_type]

    # Setup model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, model_class, device)

    # Generate predictions
    predictions = predict_from_data(model, signals, static_feat, device)

    # Extract model name from path for dynamic title and save path
    model_name = model_path.split('/')[-1].replace('.pth', '')
    title = f"Bland-Altman plot for {bp_type} - {model_name}"
    save_path = f"bland_altman_{model_name}_{bp_type}.png"
    
    # Create Bland-Altman plot
    stats = bland_altman_plot(labels, predictions, title=title, save_path=save_path)

    return labels, predictions, stats

# Define the models list with paths and classes
MODELS = [
    ("/project/zouridakis/gzlab2/demographic_models/094329/checkpoint_epoch_97.pth", MResnet50_1D, "MResNet50", "SBP"),
    ("/project/zouridakis/gzlab2/demographic_models/160257/checkpoint_epoch_95.pth", MResnet18_1D, "MResNet18", "SBP"),
    ("/project/zouridakis/gzlab2/demographic_models/091710/checkpoint_epoch_33.pth", MLeNet1d, "MLeNet", "SBP"),
    ("/project/zouridakis/gzlab2/demographic_models/235021/checkpoint_epoch_91.pth", MInception1d, "MInception", "SBP"),
    ("/project/zouridakis/gzlab2/demographic_models/002418/checkpoint_epoch_95.pth", MResnet50_1D, "MResNet50", "DBP"),
    ("/project/zouridakis/gzlab2/demographic_models/172631/checkpoint_epoch_98.pth", MResnet18_1D, "MResNet18", "DBP"),
    ("/project/zouridakis/gzlab2/demographic_models/091635/checkpoint_epoch_22.pth", MLeNet1d, "MLeNet", "DBP"),
    ("/project/zouridakis/gzlab2/demographic_models/234820/checkpoint_epoch_92.pth", MInception1d, "MInception", "DBP"),
]

# Data path
data_path = '/project/zouridakis/gzlab2/PulseDB/Subset_Files/CalFree_Test_Subset_filtered.mat'

# Run analysis for each model
for model_path, model_class, model_name, bp_type in MODELS:
    print(f"Running analysis for {model_name} {bp_type}...")
    try:
        y_true, y_pred, stats = run_bland_altman_analysis(data_path, model_path, model_class, bp_type)
    except FileNotFoundError:
        print(f"Model file not found: {model_path}. Skipping.")
    except Exception as e:
        print(f"Error running analysis for {model_name} {bp_type}: {e}")

