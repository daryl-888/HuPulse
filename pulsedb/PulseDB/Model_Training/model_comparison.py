import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from Model_Def.MResNet18 import MResnet18_1D
from Model_Def.MResNet50 import MResnet50_1D
from Model_Def.MS4 import MS4_1D
from Model_Def.Minception import MInception1d
from Model_Def.MLenet import MLeNet1d

def load_s4_model_safe(model_class, checkpoint, device):
    """Safely load S4 model with special handling"""
    model = model_class(s4_l_max=2048)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model_dict = dict(model.named_parameters())
    
    for name, param in state_dict.items():
        if name in model_dict:
            if any(x in name for x in ['kernel.B', 'kernel.P', 'kernel.w']):
                model_dict[name].data = param.clone().data
            else:
                model_dict[name].data = param.data
    
    return model

def load_model(model_path, model_class, device):
    """Load a single model"""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        if model_class.__name__ == 'MS4_1D':
            model = load_s4_model_safe(model_class, checkpoint, device)
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

def load_checkpoint_metrics(checkpoint_path):
    """Load metrics from checkpoint file"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Common keys where metrics might be stored
        if 'test_mae' in checkpoint:
            return checkpoint['test_mae']
        elif 'val_mae' in checkpoint:
            return checkpoint['val_mae']
        elif 'mae' in checkpoint:
            return checkpoint['mae']
        elif 'loss' in checkpoint:
            return checkpoint['loss']
        else:
            # If no direct metric, return a placeholder
            print(f"No standard metric found in {checkpoint_path}")
            return None
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def create_performance_comparison(checkpoint_dir, metric_name="MAE"):
    """Create bar chart comparing model performance"""
    
    # Define your models and their checkpoint files
    models = {
        'ResNet18': 'resnet18_best.pth',
        'ResNet34': 'resnet34_best.pth', 
        'ResNet50': 'resnet50_best.pth',
        'ResNet101': 'resnet101_best.pth'
    }
    
    model_names = []
    performance_values = []
    
    # Load performance metrics from each checkpoint
    for model_name, checkpoint_file in models.items():
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        if os.path.exists(checkpoint_path):
            metric_value = load_checkpoint_metrics(checkpoint_path)
            if metric_value is not None:
                model_names.append(model_name)
                performance_values.append(metric_value)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    if not performance_values:
        print("No valid metrics found in checkpoints")
        return
    
    # Create bar chart with colors matching number of models
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    model_colors = colors[:len(model_names)]  # Only use as many colors as models
    
    bars = plt.bar(model_names, performance_values, color=model_colors)
    
    # Customize the plot
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel(f'{metric_name} (mmHg)', fontsize=12)
    plt.title(f'SBP Prediction Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, performance_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\nModel Performance Summary:")
    for name, value in zip(model_names, performance_values):
        print(f"{name}: {value:.2f} {metric_name}")

# Usage
if __name__ == "__main__":
    # Set your checkpoint directory path
    checkpoint_directory = "./checkpoints"  # Update this path
    
    create_performance_comparison(checkpoint_directory, metric_name="MAE")