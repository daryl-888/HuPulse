import torch
import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
from collections import defaultdict
from scipy import stats

from Model_Def.MResNet18 import MResnet18_1D
from Model_Def.MResNet50 import MResnet50_1D
from Model_Def.MS4 import MS4_1D
from Model_Def.Minception import MInception1d
from Model_Def.MLenet import MLeNet1d

plt.rcParams.update({'font.size': 16})

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_patient_data(subset_path, model, device, label_type: str = "SBP"):
    """Extract predictions and ground truth organized by patient"""
    
    # Load the subset data
    data = loadmat(subset_path)
    subjects = data['Subset']['Subject']
    signals = data['Subset']['Signals']
    labels = data['Subset'][label_type]  # Use SBP or DBP based on label_type
    age = data['Subset']['Age']
    gender = data['Subset']['Gender']
    bmi = data['Subset']['BMI']
    
    # Group data by patient
    patient_data = defaultdict(lambda: {'predictions': [], 'ground_truth': [], 'segment_indices': []})
    
    with torch.no_grad():
        for i in range(len(subjects)):
            # Extract patient ID (remove the source identifier at the end)
            subject_id = subjects[i][0] if isinstance(subjects[i], (list, np.ndarray)) else subjects[i]
            patient_id = str(subject_id[:7])  # e.g., 'p000020' from 'p000020_0'

            # Get PPG signal (channel 1)
            signal = signals[i:i+1, 1:2, :]  # Shape: [1, 1, 1250]
            signal_tensor = torch.FloatTensor(signal).to(device)

            # Get static features (age, gender, BMI) for this segment
            age_val = float(age[i])
            bmi_val = float(bmi[i])
            gender_str = gender[i][0] if isinstance(gender[i], (list, np.ndarray)) else gender[i]
            gender_num = 1.0 if gender_str == 'M' else 0.0
            static_feat = [age_val, gender_num, bmi_val]
            static_feat_tensor = torch.FloatTensor([static_feat]).to(device)

            # Get model prediction
            prediction = model(signal_tensor, static_feat_tensor).cpu().numpy().item()
            ground_truth = labels[i]  # Updated to use dynamic labels

            # Append prediction and ground truth
            patient_data[patient_id]['predictions'].append(prediction)
            patient_data[patient_id]['ground_truth'].append(ground_truth)
            patient_data[patient_id]['segment_indices'].append(len(patient_data[patient_id]['predictions']))
    
    return patient_data

def calculate_intervals(predictions, ground_truth, interval_type='PI', confidence=0.95):
    """Calculate prediction intervals (PI) or confidence intervals (CI)"""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Residuals is the prediction errors
    residuals = predictions - ground_truth
    n = len(residuals)
    
    # Need at least 2 data points for std calculation
    if n < 2:
        return np.full(len(predictions), 0.0)
    
    # Standard deviation of residuals
    residual_std = np.std(residuals, ddof=1)
    
    # t-critical value for confidence level
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, n - 1)

    if interval_type.upper() == 'PI':
        # Prediction interval: uncertainty for individual predictions
        interval = t_critical * residual_std * np.sqrt(1 + 1/n)
    elif interval_type.upper() == 'CI':
        # Confidence interval: uncertainty for mean prediction
        interval = t_critical * residual_std / np.sqrt(n)
    else:
        raise ValueError("interval_type must be 'PI' or 'CI'")
    
    return np.full(len(predictions), interval)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_top10_segments(patient_data, interval_type='CI', figsize=(12, 6)):
    """Plot only top 10 patients with most segments"""
    
    # Get all patients and sort by segment count
    patient_list = []
    for patient_id, data in patient_data.items():
        preds = np.array(data['predictions'])
        truths = np.array(data['ground_truth'])
        if len(preds) < 2:
            continue
        patient_list.append((patient_id, data, len(preds)))
    
    # Sort by segment count (descending) and take top 10
    patient_list.sort(key=lambda x: x[2], reverse=True)
    top10_patients = patient_list[:10]
    
    # Extract data for plotting
    patient_ids = []
    mean_predictions = []
    mean_ground_truth = []
    intervals = []
    segment_counts = []

    for patient_id, data, n_segments in top10_patients:
        preds = np.array(data['predictions'])
        truths = np.array(data['ground_truth'])
        
        patient_ids.append(patient_id)
        mean_predictions.append(np.median(preds))
        mean_ground_truth.append(np.median(truths))
        segment_counts.append(n_segments)
        
        # Calculate interval
        patient_interval = calculate_intervals(preds, truths, interval_type)[0]
        intervals.append(patient_interval)

    fig, ax = plt.subplots(figsize=figsize)
    patient_numbers = range(1, 11)  # 1 to 10
    
    # Plot
    ax.plot(patient_numbers, mean_ground_truth, 'ro-', 
           label='Ground Truth', markersize=6, linewidth=2)
    ax.errorbar(patient_numbers, mean_predictions, yerr=intervals, 
               fmt='bo-', capsize=4, capthick=1, elinewidth=1, 
               ecolor='grey', label=f'Predictions (95% {interval_type})', 
               markersize=6, linewidth=2)
    
    # Add segment count labels
    for i, (patient_id, count) in enumerate(zip(patient_ids, segment_counts)):
        ax.text(i+1, mean_predictions[i] + intervals[i] + 1, 
               f'{count}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Top 10 Patients (Ranked by Segment Count)')
    ax.set_ylabel('Median SBP (mmHg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(patient_numbers)
    ax.set_xticklabels([f'{pid}\n({count})' for pid, count in zip(patient_ids, segment_counts)], 
                      rotation=45, ha='right')
    plt.tight_layout()
    return fig

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_model(model_path, model_class, device):
    """Load a single model"""
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

def load_all_experiments(models, test_subset_path, device):
    """Load all experiments and extract their predictions"""
    
    experiments_data = {}
    
    for checkpoint, model_class, model_name, label_type in models:
        exp_name = f"{model_name} {label_type}"  # e.g., "MResNet50 SBP"
        print(f"\nLoading {exp_name}...")
        model = load_model(checkpoint, model_class, device)
        
        # Extract predictions for this model
        patient_data = extract_patient_data(test_subset_path, model, device, label_type)
        experiments_data[exp_name] = patient_data
        
        print(f"  - Extracted data for {len(patient_data)} patients")
    
    return experiments_data



# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_subset_path = "/project/zouridakis/gzlab2/PulseDB/Subset_Files/CalFree_Test_Subset_filtered.mat"
    
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
    
    # Load all experiments
    print("Loading all experiments and extracting patient data...")
    experiments_data = load_all_experiments(MODELS, test_subset_path, device)
    
    print("\nAnalyzing large confidence intervals...")
    # You can now loop through experiments_data to analyze each one.
    # For simplicity, we'll analyze just the first one as before.
    first_exp_data = list(experiments_data.values())[0]
    # ci_analysis = analyze_large_cis(first_exp_data)
    
    # VISUALIZATIONS
    print("\nGenerating visualizations for all models...")
    
    # Loop through each experiment and generate a plot
    for exp_name, patient_data in experiments_data.items():
        print(f"\nCreating segment plot for: {exp_name}")
        
        # Call the plot function
        fig_top10 = plot_top10_segments(patient_data, interval_type='CI')
        
        # Dynamically set title and filename
        clean_exp_name = exp_name.replace(':', '').replace(' ', '_').replace('SBP', 'sbp').replace('DBP', 'dbp')
        fig_top10.suptitle(f'{exp_name} - Prediction Intervals', fontsize=14, y=1.02)
        
        filename = f'top10_patients_{clean_exp_name}.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close(fig_top10)  # Close the figure to free up memory
        print(f"  - Saved plot to {filename}")
    
    # The rest of your code can be kept or removed as needed.
    # The 'single patient' section can be done for each experiment if you wish,
    # or you can simply use the data from the first experiment as you did before.
    print(f"\nExtracted data for {len(first_exp_data)} patients")
    print(f"Total segments processed: {sum(len(data['predictions']) for data in first_exp_data.values())}")
    
    # Find a patient with around 20 segments for single patient plot
    target_segments = 20
    selected_patient = None
    min_diff = float('inf')
    
    for patient_id, data in first_exp_data.items():
        n_segments = len(data['predictions'])
        diff = abs(n_segments - target_segments)
        if diff < min_diff:
            min_diff = diff
            selected_patient = patient_id
    
    print(f"\nSelected patient {selected_patient} with {len(first_exp_data[selected_patient]['predictions'])} segments")

if __name__ == "__main__":
    main()