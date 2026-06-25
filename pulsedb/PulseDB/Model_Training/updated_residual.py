"""Generate combined residual analysis across multiple patients.

This script creates a single residual distribution plot combining data from
multiple selected patients to assess overall model performance.

Configuration section below lets you set:
  - MODEL_CHECKPOINT: path to the trained model (.pth)
  - MODEL_CLASS:      architecture class to instantiate
  - TEST_SUBSET_PATH: path to the test subset .mat file
  - PATIENT_INDICES:  1-based indices of patients to include (order = first
                       appearance order in the .mat file)
  - MAX_SEGMENTS:     cap on number of segments per patient (default 50)

Output: Single PNG file: combined_patients_residuals_Npatients.png
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pykeops
pykeops.clean_pykeops()
from mat73 import loadmat
from scipy import stats
from collections import defaultdict
from typing import Dict, List

from Model_Def.MResNet18 import MResnet18_1D
from Model_Def.MResNet50 import MResnet50_1D
from Model_Def.MS4 import MS4_1D
from Model_Def.Minception import MInception1d
from Model_Def.MLenet import MLeNet1d

plt.rcParams.update({'font.size': 13})

# List of (checkpoint_path, model_class, model_name, label_type) tuples
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

TEST_SUBSET_PATH = (
    "/project/zouridakis/gzlab2/PulseDB/Subset_Files/CalFree_Test_Subset_filtered.mat"
)

# 1-based indices of the patients (order preserved from data iteration)
# Options:
# PATIENT_INDICES = [1, 24, 35, 45, 54]           # Specific patients
# PATIENT_INDICES = list(range(1, 63))             # All 62 patients
# PATIENT_INDICES = "ALL"                          # Automatically include all patients
PATIENT_INDICES = "ALL"

MAX_SEGMENTS = None  # Use only first N segments per selected patient
CONFIDENCE = 0.95
N_TRAINING_SAMPLES = 81088

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== MODEL LOADING FUNCTIONS ===================== ##

def load_model(model_path, model_class, device):
    """Load a single model"""
    print(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        if model_class.__name__ == 'MS4_1D':
            print("NO S4 MODEL")
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

# ===================== CORE FUNCTIONS ===================== #

def extract_patient_data(subset_path: str, model: torch.nn.Module, label_type: str = "SBP") -> Dict[str, dict]:
    """Extract predictions and ground truths grouped by patient.

    Assumes .mat structure with keys: Subset -> Subject, Signals, SBP, DBP, Age, Gender, BMI
    """
    print(f"Loading subset file: {subset_path} for {label_type}")
    data = loadmat(subset_path)
    subjects = data["Subset"]["Subject"]
    signals = data["Subset"]["Signals"]
    labels = data["Subset"][label_type]  # Use SBP or DBP based on label_type
    age = data["Subset"]["Age"]
    gender = data["Subset"]["Gender"]
    bmi = data["Subset"]["BMI"]

    patient_data = defaultdict(lambda: {"predictions": [], "ground_truth": []})

    with torch.no_grad():
        for i in range(len(subjects)):
            subject_entry = subjects[i]
            subject_id = subject_entry[0] if isinstance(subject_entry, (list, np.ndarray)) else subject_entry
            patient_id = str(subject_id[:7])  # 'p000020' from 'p000020_0'

            signal = signals[i : i + 1, 1:2, :]  # channel 1
            signal_tensor = torch.as_tensor(signal, dtype=torch.float32, device=DEVICE)

            age_val = float(age[i])
            bmi_val = float(bmi[i])
            gender_entry = gender[i]
            gender_str = gender_entry[0] if isinstance(gender_entry, (list, np.ndarray)) else gender_entry
            gender_num = 1.0 if gender_str == "M" else 0.0
            static_feat_tensor = torch.as_tensor([[age_val, gender_num, bmi_val]], dtype=torch.float32, device=DEVICE)
            pred = model(signal_tensor, static_feat_tensor).detach().cpu().numpy().item()
            gt = float(labels[i])  # Use the selected label type

            patient_data[patient_id]["predictions"].append(pred)
            patient_data[patient_id]["ground_truth"].append(gt)

    print(f"Extracted data for {len(patient_data)} patients using {label_type}")
    return patient_data

def plot_combined_residual_distribution(patient_ids: List[str], predictions: List[float], truths: List[float], out_path: str, model_name: str = "", label_type: str = ""):
    """Create and save combined residual distribution plots for multiple patients."""
    preds = np.array(predictions)
    gts = np.array(truths)

    # Calculate residuals
    residuals = gts - preds  # Actual - Predicted
    
    if len(residuals) < 3:
        print(f"Not enough data points for residual analysis ({len(residuals)} points)")
        return
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(12, 8))
    
    # 1. Histogram with normal overlay
    ax1 = plt.subplot(1, 1, 1)
    n_bins = min(30, max(10, len(residuals) // 20))  # More bins for larger dataset
    counts, bins, _ = ax1.hist(residuals, bins=n_bins, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    
    # Overlay normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    ax1.set_xlabel('Residuals (Actual - Predicted)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Residual Distribution - {model_name} ({label_type})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Statistical test results
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    # Compile statistics text
    patient_list = ", ".join(patient_ids[:5])  # Show first 5 patient IDs
    if len(patient_ids) > 5:
        patient_list += f" ... (+{len(patient_ids)-5} more)"
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved combined residual plot: {out_path}")


# ===================== MAIN FLOW ===================== #

def main():
    print("Starting combined residual analysis across multiple patients and models...")
    
    # Loop over each model
    for checkpoint, model_class, model_name, label_type in MODELS:
        print(f"\nProcessing model: {model_name} ({label_type})")
        model = load_model(checkpoint, model_class, DEVICE)
        patient_data = extract_patient_data(TEST_SUBSET_PATH, model, label_type)

        # Preserve insertion order (Python 3.7+ dict) for indexing.
        ordered_patients = list(patient_data.keys())
        total_patients = len(ordered_patients)
        print(f"Total patients detected: {total_patients}")

        # Handle different PATIENT_INDICES formats
        if PATIENT_INDICES == "ALL":
            selected_indices = list(range(1, total_patients + 1))
            print(f"Using ALL patients: indices 1-{total_patients}")
        else:
            selected_indices = PATIENT_INDICES
            print(f"Using selected patients: {selected_indices}")

        # Collect data from all selected patients for this model
        all_predictions = []
        all_ground_truth = []
        included_patient_ids = []

        for idx in selected_indices:
            if idx < 1 or idx > total_patients:
                print(f"Patient index {idx} out of range (1..{total_patients}); skipping.")
                continue
            
            patient_id = ordered_patients[idx - 1]
            preds_full = patient_data[patient_id]["predictions"]
            gts_full = patient_data[patient_id]["ground_truth"]

            # Limit to first MAX_SEGMENTS segments
            preds = preds_full[:MAX_SEGMENTS]
            gts = gts_full[:MAX_SEGMENTS]

            all_predictions.extend(preds)
            all_ground_truth.extend(gts)
            included_patient_ids.append(patient_id)
            print(f"  Added patient {idx} ({patient_id}): {len(preds)} segments")

        if len(all_predictions) == 0:
            print(f"No valid patients for {model_name} ({label_type}). Skipping.")
            continue

        # Generate residual analysis for this model
        output_file = f"residuals_{model_name}_{label_type}_{len(included_patient_ids)}patients.png"
        plot_combined_residual_distribution(included_patient_ids, all_predictions, all_ground_truth, output_file, model_name, label_type)

        print(f"Done for {model_name} ({label_type}). Combined analysis includes {len(all_predictions)} total data points from {len(included_patient_ids)} patients.")
    
    print("\nAll models processed.")


if __name__ == "__main__":
    main()