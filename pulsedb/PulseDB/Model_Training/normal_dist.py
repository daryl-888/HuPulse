import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat

def load_data(filepath):
    data = loadmat(filepath)['Subset']
    return {
        'gender': np.array(data['Gender']).flatten(),
        'sbp': data['SBP'].flatten(),
        'dbp': data['DBP'].flatten()
    }

# Load only cleaned datasets
data_folder = '/project/zouridakis/gzlab2/PulseDB/Subset_Files/'
files = ['Train_Subset_filtered.mat', 'CalBased_Test_Subset_filtered.mat', 'CalFree_Test_Subset_filtered.mat']

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, filename in enumerate(files):
    clean = load_data(data_folder + filename)
    dataset_name = filename.split('_')[0]
    
    # Gender distribution
    gender_counts = np.unique(clean['gender'], return_counts=True)
    male_count = np.sum(clean['gender'] == 'M')
    female_count = np.sum(clean['gender'] == 'F')
    axes[0, i].bar(['Male', 'Female'], [male_count, female_count], 
              color='red', alpha=0.7)
    axes[0, i].set_title(f'{dataset_name} - Gender (Filtered)')
    axes[0, i].set_ylabel('Segment Count')
    
    # SBP distribution
    axes[1, i].hist(clean['sbp'], bins=20, color='red', alpha=0.7)
    axes[1, i].set_title(f'{dataset_name} - SBP (Filtered)')
    axes[1, i].set_ylabel('Segment Count')
    axes[1, i].set_xlabel('SBP (mmHg)')
    
    # DBP distribution
    axes[2, i].hist(clean['dbp'], bins=20, color='red', alpha=0.7)
    axes[2, i].set_title(f'{dataset_name} - DBP (Filtered)')
    axes[2, i].set_ylabel('Segment Count')
    axes[2, i].set_xlabel('DBP (mmHg)')

plt.tight_layout()
plt.savefig('normal_dist_filtered.png')

# Print summary stats for filtered only
for filename in files:
    clean = load_data(data_folder + filename)
    dataset_name = filename.split('_')[0]
    
    print(f"\n{dataset_name} (Filtered):")
    print(f"  Total samples: {len(clean['gender'])}")
    print(f"  Male: {np.sum(clean['gender'] == 'M')}")
    print(f"  Female: {np.sum(clean['gender'] == 'F')}")
    print(f"  SBP: {clean['sbp'].mean():.1f}±{clean['sbp'].std():.1f}")
    print(f"  DBP: {clean['dbp'].mean():.1f}±{clean['dbp'].std():.1f}")


# Combine all filtered datasets
all_datasets = []
for filename in files:
    all_datasets.append(load_data(data_folder + filename))

# Combine data
combined = {
    'gender': np.concatenate([d['gender'] for d in all_datasets]),
    'sbp': np.concatenate([d['sbp'] for d in all_datasets]),
    'dbp': np.concatenate([d['dbp'] for d in all_datasets])
}

# Plot combined distribution
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Gender
male_count = np.sum(combined['gender'] == 'M')
female_count = np.sum(combined['gender'] == 'F')
axes2[0].bar(['Male', 'Female'], [male_count, female_count], color='green', alpha=0.7)
axes2[0].set_title('Total Combined - Gender (Filtered)')
axes2[0].set_ylabel('Count')

# SBP
axes2[1].hist(combined['sbp'], bins=30, color='green', alpha=0.7)
axes2[1].set_title('Total Combined - SBP (Filtered)')
axes2[1].set_ylabel('Segment Count')
axes2[1].set_xlabel('SBP (mmHg)')

# DBP
axes2[2].hist(combined['dbp'], bins=30, color='green', alpha=0.7)
axes2[2].set_title('Total Combined - DBP (Filtered)')
axes2[2].set_ylabel('Segment Count')
axes2[2].set_xlabel('DBP (mmHg)')

plt.tight_layout()
plt.savefig('normal_dist_combined_filtered.png')

# Print combined stats
print(f"\nCOMBINED TOTAL (Filtered):")
print(f"  Total samples: {len(combined['gender'])}")
print(f"  Male: {male_count}")
print(f"  Female: {female_count}")
print(f"  SBP: {combined['sbp'].mean():.1f}±{combined['sbp'].std():.1f}")
print(f"  DBP: {combined['dbp'].mean():.1f}±{combined['dbp'].std():.1f}")