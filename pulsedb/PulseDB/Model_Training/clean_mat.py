import numpy as np
from mat73 import loadmat
import hdf5storage
import h5py

def save_matlab_v73_fixed(output_path, data_dict):
    """
    Fixed version that handles object arrays properly to avoid address overflow
    while maintaining MATLAB cell array compatibility
    """
    
    # Method 1: Use hdf5storage with specific options to handle large cell arrays
    print("  → Attempting Method 1: hdf5storage with cell array options...")
    try:
        hdf5storage.savemat(output_path, data_dict, 
                           format='7.3', 
                           matlab_compatible=True, 
                           compress=True,
                           truncate_existing=True,
                           store_python_metadata=False,  # Reduces metadata overhead
                           action_for_matlab_incompatible='error')
        print("  ✓ Method 1 successful!")
        return True
        
    except Exception as e1:
        print(f"  ✗ Method 1 failed: {e1}")
        
        # Method 2: Try with different chunking strategy
        print("  → Attempting Method 2: Custom chunking...")
        try:
            with h5py.File(output_path, 'w', driver='core', backing_store=True) as f:
                # Use core driver with backing store for better memory management
                subset_group = f.create_group('Subset')
                
                for key, value in data_dict['Subset'].items():
                    if isinstance(value, np.ndarray) and value.dtype == object:
                        # For cell arrays, create with smaller chunks
                        if key in ['Subject', 'Gender']:
                            # Convert to bytes for HDF5 compatibility but maintain cell structure
                            str_list = []
                            for item in value.flatten():
                                if hasattr(item, '__getitem__') and len(item) > 0:
                                    str_list.append(str(item[0]).encode('utf-8'))
                                else:
                                    str_list.append(str(item).encode('utf-8'))
                            
                            # Create variable-length string dataset (maintains cell-like behavior)
                            dt = h5py.special_dtype(vlen=bytes)
                            dset = subset_group.create_dataset(
                                key, 
                                (len(str_list),), 
                                dtype=dt,
                                chunks=(min(1000, len(str_list)),),  # Small chunks for cell arrays
                                compression='gzip'
                            )
                            dset[:] = str_list
                            
                            # Add MATLAB cell array attributes
                            dset.attrs['MATLAB_class'] = b'cell'
                            dset.attrs['MATLAB_size'] = np.array([len(str_list), 1])
                        else:
                            subset_group.create_dataset(key, data=value)
                    else:
                        # Regular numeric arrays
                        if hasattr(value, 'size') and value.size > 10000:
                            chunks = True
                        else:
                            chunks = None
                        subset_group.create_dataset(key, data=value, chunks=chunks, compression='gzip')
            
            print("  ✓ Method 2 successful!")
            return True
            
        except Exception as e2:
            print(f"  ✗ Method 2 failed: {e2}")
        
        # Method 2: Use h5py with explicit dataset creation
        print("  → Attempting Method 2: Direct h5py...")
        try:
            with h5py.File(output_path, 'w') as f:
                subset_group = f.create_group('Subset')
                
                for key, value in data_dict['Subset'].items():
                    if isinstance(value, np.ndarray):
                        if value.dtype == object and key in ['Subject', 'Gender']:
                            # Handle cell arrays as variable-length strings
                            str_data = [str(item[0]) if hasattr(item, '__getitem__') and len(item) > 0 
                                      else str(item) for item in value.flatten()]
                            
                            # Create variable-length string dataset
                            dt = h5py.special_dtype(vlen=str)
                            dset = subset_group.create_dataset(key, (len(str_data),), dtype=dt)
                            dset[:] = str_data
                            
                        else:
                            # Handle regular arrays with proper chunking
                            if value.size > 10000:  # Use chunking for large arrays
                                chunks = True
                            else:
                                chunks = None
                            
                            subset_group.create_dataset(key, data=value, 
                                                     chunks=chunks, compression='gzip')
                    else:
                        subset_group.create_dataset(key, data=value)
            
            print("  ✓ Method 2 successful!")
            return True
            
        except Exception as e2:
            print(f"  ✗ Method 2 failed: {e2}")
            
            # Method 3: Flatten object arrays completely
            print("  → Attempting Method 3: Flatten object arrays...")
            try:
                flattened_data = {}
                for key, value in data_dict['Subset'].items():
                    if isinstance(value, np.ndarray) and value.dtype == object:
                        # Convert to 1D string array
                        flat_strings = []
                        for item in value.flatten():
                            if hasattr(item, '__getitem__') and len(item) > 0:
                                flat_strings.append(str(item[0]))
                            else:
                                flat_strings.append(str(item))
                        flattened_data[key] = np.array(flat_strings)
                    else:
                        flattened_data[key] = value
                
                hdf5storage.savemat(output_path, {'Subset': flattened_data}, 
                                   format='7.3', matlab_compatible=True, compress=True)
                print("  ✓ Method 3 successful!")
                return True
                
            except Exception as e3:
                print(f"  ✗ Method 3 failed: {e3}")
                print(f"  ✗ All methods failed. Unable to save {output_path}")
                return False

def clean_subset(input_path, output_path, name):
    print(f"\n{name}:")
    
    try:
        data = loadmat(input_path)['Subset']
    except Exception as e:
        print(f"  ✗ Failed to load {input_path}: {e}")
        return 0, 0, 0, 0
    
    # Extract data with error handling
    try:
        subjects = np.array(data['Subject']).squeeze()
        signals = data['Signals']
        sbp, dbp = data['SBP'].squeeze(), data['DBP'].squeeze()
        age = data['Age'].squeeze()
        gender = np.array(data['Gender']).squeeze()
        height, weight, bmi = data['Height'].squeeze(), data['Weight'].squeeze(), data['BMI'].squeeze()
    except Exception as e:
        print(f"  ✗ Error extracting data: {e}")
        return 0, 0, 0, 0
    
    orig_count = len(subjects)
    orig_subjects = len(set(subjects))
    
    # Remove subjects with NaN BMI
    valid_subjects = set(subjects[~np.isnan(bmi)].tolist())
    mask = np.array([s in valid_subjects for s in subjects])
    
    # Apply filters
    mask &= (age >= 18) & (age <= 65)
    mask &= (bmi >= 18.5) & (bmi <= 25.0)
    mask &= (sbp >= 90) & (sbp <= 130)
    mask &= (dbp >= 60) & (dbp <= 85)
    
    final_count = np.sum(mask)
    final_subjects = len(set(subjects[mask].tolist()))
    
    print(f"  Segments: {orig_count} → {final_count} ({final_count/orig_count*100:.1f}%)")
    print(f"  Subjects: {orig_subjects} → {final_subjects}")
    
    if final_count == 0:
        print(f"  ⚠ No data remaining after filtering!")
        return orig_count, 0, orig_subjects, 0  # Return 0 for final counts
    
    # Create proper object arrays for cell arrays - but handle them better
    subject_cell = np.empty((final_count, 1), dtype=object)
    gender_cell = np.empty((final_count, 1), dtype=object)
    
    for i, (s, g) in enumerate(zip(subjects[mask], gender[mask])):
        subject_cell[i, 0] = s
        gender_cell[i, 0] = g
    
    cleaned = {
        'Subject': subject_cell,
        'Signals': signals[mask],
        'SBP': sbp[mask].reshape(-1, 1),
        'DBP': dbp[mask].reshape(-1, 1),
        'Age': age[mask].reshape(-1, 1),
        'Gender': gender_cell,
        'Height': height[mask].reshape(-1, 1),
        'Weight': weight[mask].reshape(-1, 1),
        'BMI': bmi[mask].reshape(-1, 1)
    }
    
    # Debug: Print data shapes and types
    print(f"  Data shapes: Subjects={subject_cell.shape}, Signals={signals[mask].shape}")
    print(f"  Object array sample: Subject[0]={subject_cell[0,0]}, Gender[0]={gender_cell[0,0]}")
    
    # Try to save with improved object array handling
    print(f"  Attempting to save {final_count} samples...")
    success = save_matlab_v73_fixed(output_path, {'Subset': cleaned})
    
    if success:
        print(f"  ✓ Successfully saved {name}")
        return orig_count, final_count, orig_subjects, final_subjects
    else:
        print(f"  ✗ Failed to save {name}")
        return orig_count, 0, orig_subjects, 0  # Return 0 for final counts if save failed

# Process files
data_folder = '/project/zouridakis/gzlab2/PulseDB/Subset_Files/'
files = [
    ('Train_Subset.mat', 'Train_Subset_cleaned.mat', 'Train'),
    ('CalBased_Test_Subset.mat', 'CalBased_Test_Subset_cleaned.mat', 'CalBased'),
    ('CalFree_Test_Subset.mat', 'CalFree_Test_Subset_cleaned.mat', 'CalFree'),
    ('AAMI_Test_Subset.mat', 'AAMI_Test_Subset_cleaned.mat', 'AAMI')
]

total_orig_seg = total_final_seg = total_orig_subj = total_final_subj = 0

for input_file, output_file, name in files:
    try:
        orig_seg, final_seg, orig_subj, final_subj = clean_subset(
            data_folder + input_file, data_folder + output_file, name)
        total_orig_seg += orig_seg
        total_final_seg += final_seg
        total_orig_subj += orig_subj
        total_final_subj += final_subj
    except Exception as e:
        print(f"Error: {name} - {e}")

# Fixed division by zero error
print(f"\n" + "="*50)
print(f"SUMMARY:")
if total_orig_seg > 0:
    print(f"Total segments: {total_orig_seg} → {total_final_seg} ({total_final_seg/total_orig_seg*100:.1f}%)")
else:
    print(f"Total segments: 0 → {total_final_seg} (No successful processing)")

if total_orig_subj > 0:
    print(f"Total subjects: {total_orig_subj} → {total_final_subj}")
else:
    print(f"Total subjects: 0 → {total_final_subj} (No successful processing)")
print(f"="*50)