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

def extract_patient_data(subset_path, model, device):
    """Extract predictions and ground truth organized by patient"""
    
    # Load the subset data
    data = loadmat(subset_path)
    subjects = data['Subset']['Subject']
    signals = data['Subset']['Signals']
    sbp_labels = data['Subset']['SBP']
    # dbp_labels = data['Subset']['DBP']
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
            ground_truth = sbp_labels[i]

            # Append prediction and ground truth
            patient_data[patient_id]['predictions'].append(prediction)
            patient_data[patient_id]['ground_truth'].append(ground_truth)
            patient_data[patient_id]['segment_indices'].append(len(patient_data[patient_id]['predictions']))
    
    return patient_data

def calculate_prediction_intervals(predictions, ground_truth, confidence=0.95):
    """Calculate prediction intervals based on model residuals"""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Residuals is the prediction errors, which is the predictions - ground truth
    residuals = predictions - ground_truth

    # Get the number of residuals which is the length of the array
    n = len(residuals)
    
    # Need at least 2 data points for std calculation
    if n < 2:
        return np.full(len(predictions), 0.0)  # Return zero interval for single points
    
    # Standard deviation of residuals (model's typical error)
    residual_std = np.std(residuals, ddof=1)
    # print('residual std:', residual_std)
    
    # t-critical value for confidence level
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, n - 1)
    # print('t critical:', t_critical)


    prediction_interval = t_critical * residual_std / np.sqrt(n)
    
    return np.full(len(predictions), prediction_interval)

def create_single_patient_time_series(patient_id, patient_data, figsize=(12, 8)):
    """Create a detailed time series plot for a single patient"""
    
    data = patient_data[patient_id]
    predictions = np.array(data['predictions'])
    ground_truth = np.array(data['ground_truth'])
    n_segments = len(predictions)
    segment_numbers = np.arange(1, n_segments + 1)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Calculate confidence intervals
    confidence_intervals = calculate_prediction_intervals(predictions, ground_truth)
    
    # Plot ground truth
    ax.plot(segment_numbers, ground_truth, 'ro-', 
           label='Ground Truth', markersize=8, linewidth=3, alpha=0.9)
    
    # Plot predictions with error bars
    ax.errorbar(segment_numbers, predictions, 
               yerr=confidence_intervals,
               fmt='bo-',
               label='Predictions (95% CI)',
               alpha=0.9, 
               capsize=5,
               capthick=2,
               elinewidth=2,
               markersize=8,
               linewidth=3,
               ecolor='lightsteelblue')
    
    # Calculate metrics
    r_squared = stats.pearsonr(ground_truth, predictions)[0] ** 2
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    mae = np.mean(np.abs(predictions - ground_truth))
    
    ax.set_xlabel('Segment Number', fontsize=14)
    ax.set_ylabel('SBP (mmHg)', fontsize=14)
    ax.set_title(f'Patient {patient_id} - Blood Pressure Prediction Over Segments\n'
                f'Total Segments: {n_segments}', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Set x-axis to show all segment numbers if reasonable, otherwise subsample
    if n_segments <= 20:
        ax.set_xticks(segment_numbers)
    else:
        ax.set_xticks(segment_numbers[::max(1, n_segments//20)])
    
    plt.tight_layout()
    return fig

def plot_patient_mean_predictions(patient_data):
    """Plot mean prediction vs mean ground truth for each patient with prediction intervals."""
    patient_ids = []
    mean_predictions = []
    mean_ground_truth = []
    intervals = []

    for patient_id, data in patient_data.items():
        preds = np.array(data['predictions'])
        truths = np.array(data['ground_truth'])
        if len(preds) == 0:
            continue
        patient_ids.append(patient_id)
        mean_predictions.append(np.mean(preds))
        mean_ground_truth.append(np.mean(truths))
        # Calculate interval for the mean
        interval = calculate_prediction_intervals(preds, truths)[0] / np.sqrt(len(preds))
        intervals.append(interval)

    fig, ax = plt.subplots(figsize=(12, 6))
    patient_numbers = range(1, len(patient_ids) + 1)
    # Replace the errorbar and plot lines with:
    ax.plot(patient_numbers, mean_ground_truth, 'ro-', label='Ground Truth', markersize=6, linewidth=2)
    ax.errorbar(patient_numbers, mean_predictions, yerr=intervals, 
           fmt='bo-', capsize=4, capthick=1, elinewidth=1, 
           ecolor='grey', label='Predictions (95% CI)', 
           markersize=6, linewidth=2)
    ax.set_xlabel('Patient Number')
    ax.set_ylabel('Mean DBP (mmHg)')
    ax.set_title('Mean DBP Prediction per Patient (with 95% CI)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('patient_mean_predictions_all_patients_dbp.png', dpi=300, bbox_inches='tight')

def plot_all_experiments(experiments_data, figsize=(15, 8)):
    """Plot all 5 experiments in a grid layout suitable for research paper"""
    
    # Create 2x3 grid (top row: 3 plots, bottom row: 2 plots centered)
    fig = plt.figure(figsize=figsize)
    
    # Define subplot positions for better layout
    positions = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 5), (2, 3, 6)]
    
    for idx, (exp_name, patient_data) in enumerate(experiments_data.items()):
        ax = plt.subplot(*positions[idx])
        
        # Extract data for this experiment
        patient_ids = []
        mean_predictions = []
        mean_ground_truth = []
        intervals = []

        for patient_id, data in patient_data.items():
            preds = np.array(data['predictions'])
            truths = np.array(data['ground_truth'])
            if len(preds) == 0:
                continue
            patient_ids.append(patient_id)
            mean_predictions.append(np.mean(preds))
            mean_ground_truth.append(np.mean(truths))
            interval = calculate_prediction_intervals(preds, truths)[0] / np.sqrt(len(preds))
            print(f"Patient {patient_id}: Mean Prediction={mean_predictions[-1]:.2f}, ")
            print(interval)
            print(preds)
            # break #for prediction interval
            intervals.append(interval)

        patient_numbers = range(1, len(patient_ids) + 1)
        
        # Plot
        ax.plot(patient_numbers, mean_ground_truth, 'ro-', 
               label='Ground Truth' if idx == 0 else "", 
               markersize=3, linewidth=1.5, alpha=0.8)
        ax.errorbar(patient_numbers, mean_predictions, yerr=intervals, 
                   fmt='bo-', capsize=4, capthick=1, elinewidth=1, 
                   ecolor='grey', 
                   label='Predictions', 
                   markersize=6, linewidth=2, alpha=0.8)
        
        ax.set_title(f'{exp_name}', fontsize=10, pad=5)
        ax.set_xlabel('Patient Number', fontsize=9)
        ax.set_ylabel('Mean DBP (mmHg)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Only show legend on first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    return fig

def load_s4_model_safe(model_class, checkpoint, device):
    """Safely load S4 model with special handling for problematic tensors"""

    model = model_class(s4_l_max=2048)  # Match your training config
    
    # Get the state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Bypass load_state_dict entirely - manually set parameters
    model_dict = dict(model.named_parameters())
    
    for name, param in state_dict.items():
        if name in model_dict:
            # Direct parameter assignment, cloning problematic tensors
            if any(x in name for x in ['kernel.B', 'kernel.P', 'kernel.w']):
                model_dict[name].data = param.clone().data
            else:
                model_dict[name].data = param.data
    
    return model

def load_and_extract_all_experiments(model_paths, test_subset_path, device):
    """Load all 5 models and extract their predictions"""
    
    experiments_data = {}
    
    for exp_name, model_info in model_paths.items():
        model_path = model_info['path']
        model_class = model_info['model_class']
        
        print(f"Loading {exp_name} from {model_path}")
        print(f"  - Model class: {model_class.__name__}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            print(f"  - Checkpoint is a dictionary with keys: {list(checkpoint.keys())}")
            
            # Special handling for S4 models
            if model_class.__name__ == 'MS4_1D':
                print(f"  - Using special S4 loading method")
                try:
                    model = load_s4_model_safe(model_class, checkpoint, device)
                    print(f"  - S4 model loaded successfully")
                except Exception as e:
                    print(f"  - Error loading S4 model: {e}")
                    continue
            else:
                # Regular loading for other models
                # Try different common keys where the model might be stored
                if 'model' in checkpoint:
                    model = checkpoint['model']
                    print(f"  - Using 'model' key")
                elif 'model_state_dict' in checkpoint:
                    model = model_class()
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  - Using 'model_state_dict' key")
                elif 'state_dict' in checkpoint:
                    model = model_class()
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"  - Using 'state_dict' key")
                elif 'net' in checkpoint:
                    model = checkpoint['net']
                    print(f"  - Using 'net' key")
                else:
                    # If none of the common keys work, try to create model and load the entire dict
                    try:
                        model = model_class()
                        model.load_state_dict(checkpoint)
                        print(f"  - Loaded entire checkpoint as state_dict")
                    except Exception as e:
                        print(f"  - Error loading checkpoint: {e}")
                        print(f"  - Available keys: {list(checkpoint.keys())}")
                        continue
        else:
            # Checkpoint is already a model object
            model = checkpoint
            print(f"  - Checkpoint is a model object")
        
        model.to(device)
        model.eval()
        
        # Extract predictions for this model
        patient_data = extract_patient_data(test_subset_path, model, device)
        experiments_data[exp_name] = patient_data
        
        print(f"  - Extracted data for {len(patient_data)} patients")
    
    return experiments_data

def main():
    # Load your trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract patient data from test set
    print("Loading test data and extracting patient predictions...")
    test_subset_path = "/project/zouridakis/gzlab2/PulseDB/Subset_Files/CalFree_Test_Subset_filtered.mat"
    
    # print(f"Extracted data for {len(patient_data)} patients")
    # print(f"Total segments processed: {sum(len(data['predictions']) for data in patient_data.values())}")
    
    # # Print summary of patient segments
    # sorted_patients = sorted(patient_data.items(), key=lambda x: len(x[1]['predictions']), reverse=True)
    # print(f"\nPatient segment distribution:")
    # print(f"Most segments: {len(sorted_patients[0][1]['predictions'])} (Patient {sorted_patients[0][0]})")
    # print(f"Least segments: {len(sorted_patients[-1][1]['predictions'])} (Patient {sorted_patients[-1][0]})")
    # avg_segments = sum(len(data['predictions']) for data in patient_data.values()) / len(patient_data)
    # print(f"Average segments per patient: {avg_segments:.1f}")
    
    # Plot mean predictions vs ground truth for all patients
    # print("\nCreating patient mean predictions plot...")
    # plot_patient_mean_predictions(patient_data)
    # print("Saved: patient_mean_predictions_all_patients_dbp.png")

    # Define your 5 model paths
    model_paths = {
        'Exp1: MResNet50 SBP': {
            'path': "/project/zouridakis/gzlab2/demographic_models/094329/checkpoint_epoch_4.pth",
            'model_class': MResnet50_1D
        },
        # 'Exp2: MResNet18 SBP': {
        #     'path': "/project/zouridakis/gzlab2/demographic_models/160257/checkpoint_epoch_95.pth",
        #     'model_class': MResnet18_1D
        # },
        # 'Exp3: LeNet1D SBP': {
        #     'path': "/project/zouridakis/gzlab2/demographic_models/091710/checkpoint_epoch_8.pth",
        #     'model_class': MLeNet1d
        # },
        # 'Exp4: MInception1D SBP': {
        #     'path': "/project/zouridakis/gzlab2/demographic_models/235021/checkpoint_epoch_28.pth",
        #     'model_class': MInception1d
        # },
        # 'Exp5: S4 SBP': {
        #     'path': "/project/zouridakis/gzlab2/demographic_models/091857/checkpoint_epoch_38.pth",
        #     'model_class': MS4_1D
        # }
    }
    
    # Load all experiments
    experiments_data = load_and_extract_all_experiments(model_paths, test_subset_path, device)
    
    # Create comparison plot
    fig = plot_all_experiments(experiments_data)
    plt.savefig('confidence_interval.png', dpi=300, bbox_inches='tight')
    print("Saved: confidence_interval.png")



if __name__ == "__main__":
    main()