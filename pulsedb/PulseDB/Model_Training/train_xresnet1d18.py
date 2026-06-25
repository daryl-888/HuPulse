import torch
import torch.utils.data as data
import random
import numpy as np
import os
from mat73 import loadmat
from Model_Def.Trainer import Model_Trainer
from xresnet1d18 import XResNet1D18
# from torch.optim.lr_scheduler import CosineAnnealingLR


print('Torch available or not:', torch.cuda.is_available())
print('Number of GPUs available:', torch.cuda.device_count())

def Seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Dataset(data.Dataset):
    def __init__(self, Input, Label):
        self.Input = Input
        self.Label = Label

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        return self.Input[idx, :], self.Label[[idx]]


def Build_Dataset(Path, Label):
    Data = loadmat(Path)
    # Get the PPG channel (index 1) as input
    return Dataset(Data['Subset']['Signals'][:, 1:2, :], Data['Subset'][Label])


# Data paths - update with your folder location
data_folder = '/project/zouridakis/gzlab2/PulseDB/Subset_Files/'
Train_File = data_folder+'Train_Subset.mat'
Test_CalBased_File = data_folder+'CalBased_Test_Subset.mat'
Test_CalFree_File = data_folder+'CalFree_Test_Subset.mat'


# Training model for estimating SBP. Replace 'SBP' with 'DBP' to train model for DBP.
bp_type = 'SBP'
Train_Data = Build_Dataset(Train_File, bp_type)
Test_CalBased_Data = Build_Dataset(Test_CalBased_File, bp_type)
Test_CalFree_Data = Build_Dataset(Test_CalFree_File, bp_type)

if __name__ == '__main__':
    # Initialize model
    Seed(6)
    model = XResNet1D18(num_classes=1)
    Seed(6)
    print('Training custom model with full dataset delta 8 lr 1e-4')
    Settings = {
    'BP_optimizer': 'torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)',
    'trainer': 'Model_Trainer(model, torch.nn.HuberLoss(delta=8.0), BP_optimizer, device, Settings, batch_size=64, num_epochs=100, save_states=True, save_final=True)'
}
    # 'BP_scheduler': 'torch.optim.lr_scheduler.CosineAnnealingLR(BP_optimizer, T_max=100, eta_min=1e-6)',

    # Setup training device
    torch.cuda.empty_cache()
    
    # Use cuda if available, otherwise CPU
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"Primary GPU: {torch.cuda.get_device_name(0)}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create the model
    model.to(device)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    
    # Instantiate optimizer and model trainer
    BP_optimizer = eval(Settings['BP_optimizer'])
    # BP_scheduler = eval(Settings['BP_scheduler'])
    model_trainer = eval(Settings['trainer'])
    # model_trainer.scheduler = BP_scheduler  
    
    # Set the training set and the two setting set under comparison
    model_trainer.Set_Dataset(Train_Data, {
                              'Test_CalBased': Test_CalBased_Data, 'Test_CalFree': Test_CalFree_Data})
    
    # Train the model
    model_trainer.Train_Model()