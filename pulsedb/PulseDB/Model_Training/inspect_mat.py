from mat73 import loadmat

mat_path = "/project/zouridakis/gzlab2/PulseDB/Subset_Files/CalFree_Test_Subset_filtered.mat"
data = loadmat(mat_path)
subset = data['Subset']

for key in subset:
    arr = subset[key]
    print(f"\nField: {key}")
    print(f"  Type: {type(arr)}")
    try:
        print(f"  Shape: {arr.shape}")
    except AttributeError:
        print("  No shape attribute (likely a list or scalar)")
    # Print first 3 values for inspection
    try:
        print(f"  First 3 values: {arr[:3]}")
    except Exception as e:
        print(f"  Could not print first 3 values: {e}")

# For unique patient IDs
subjects = subset['Subject']
print(f"\nNumber of segments: {len(subjects)}")
unique_patients = set(str(s[:7]) for s in subjects)
print(f"Number of unique patients: {len(unique_patients)}")
print(f"First 5 unique patient IDs: {list(unique_patients)[:5]}")