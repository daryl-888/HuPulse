import torch
import torch.nn as nn
import torch.utils.data as data
import random
import numpy as np
from mat73 import loadmat
from Model_Def.Trainerv2 import Model_Trainer
from Model_Def import Mtransformerv2, Mtransformerv3, Mtransformerv4,Mtransformerv3_noshape

def encode_gender(gender_arr):
    # ??gender_arr??????,???['M', 'F', 'M', ...]
    return np.array([1 if str(g).upper().startswith('M') else 0 for g in gender_arr], dtype=np.float32)

def Seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Dataset(data.Dataset):
    def __init__(self, Input, Age, BMI, Gender, Label):
        self.Input = Input              # (N, 1, seq_len)
        self.Age = Age                 # (N,)
        self.BMI = BMI                 # (N,)
        self.Gender = Gender           # (N,)  # ????0/1
        self.Label = Label             # (N,)

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        static_feat = np.array([self.Age[idx], self.BMI[idx], self.Gender[idx]], dtype=np.float32)
        return (self.Input[idx, :].astype(np.float32), static_feat), self.Label[idx]  # x, y

# def Build_Dataset(Path, Label, show_info=True):
#     Data = loadmat(Path)
#     signals = np.expand_dims(Data['Subset']['Signals'][:, 1, :], axis=1)    # (N, 1, seq_len)
#     age = np.array(Data['Subset']['Age'], dtype=np.float32).flatten()        # (N,)
#     bmi = np.array(Data['Subset']['BMI'], dtype=np.float32).flatten()        # (N,)
#     gender_raw = Data['Subset']['Gender']                                    # (N,)
#     gender = encode_gender(gender_raw)
#     label = np.array(Data['Subset'][Label], dtype=np.float32).flatten()      # (N,)
#
#     # ??nan?inf,?????????
#     # ???????????(??+??+label)
#     N = len(age)
#     before_num = N
#     mask = np.ones(N, dtype=bool)
#     for i in range(N):
#         # ??????
#         features = [age[i], bmi[i], gender[i]]
#         # ?????????nan/inf??
#         if (
#             np.any(np.isnan(features)) or np.any(np.isinf(features)) or
#             np.any(np.isnan(signals[i])) or np.any(np.isinf(signals[i]))
#         ):
#             mask[i] = False
#     # ??????
#     signals = signals[mask]
#     age = age[mask]
#     bmi = bmi[mask]
#     gender = gender[mask]
#     label = label[mask]
#     after_num = len(age)
#
#     if show_info:
#         print(f"{Path}: Before removing nan/inf: {before_num}, After: {after_num}")
#
#     return Dataset(signals, age, bmi, gender, label)


def z_score_normalize(array):
    mean = np.mean(array)
    std = np.std(array)
    if std == 0:
        return np.zeros_like(array)  # 避免除零
    return (array - mean) / std

def Build_Dataset(Path, Label, show_info=True):
    Data = loadmat(Path)
    signals = np.expand_dims(Data['Subset']['Signals'][:, 1, :], axis=1)    # (N, 1, seq_len)
    age = np.array(Data['Subset']['Age'], dtype=np.float32).flatten()        # (N,)
    bmi = np.array(Data['Subset']['BMI'], dtype=np.float32).flatten()        # (N,)
    gender_raw = Data['Subset']['Gender']                                    # (N,)
    gender = encode_gender(gender_raw)
    label = np.array(Data['Subset'][Label], dtype=np.float32).flatten()      # (N,)

    # 过滤 nan/inf
    N = len(age)
    before_num = N
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        features = [age[i], bmi[i], gender[i]]
        if (
            np.any(np.isnan(features)) or np.any(np.isinf(features)) or
            np.any(np.isnan(signals[i])) or np.any(np.isinf(signals[i]))
        ):
            mask[i] = False
    signals = signals[mask]
    age = age[mask]
    bmi = bmi[mask]
    gender = gender[mask]
    label = label[mask]
    after_num = len(age)

    # === z-score 标准化（只对连续变量） ===
    age = z_score_normalize(age)
    bmi = z_score_normalize(bmi)
    # 性别是离散特征，不做 z-score

    if show_info:
        print(f"{Path}: Before removing nan/inf: {before_num}, After: {after_num}")
        # print(f"Age: mean={np.mean(age):.4f}, std={np.std(age):.4f}")
        # print(f"BMI: mean={np.mean(bmi):.4f}, std={np.std(bmi):.4f}")

    return Dataset(signals, age, bmi, gender, label)




data_folder =  "/home/yshen28/PPGdata/"
Train_File = data_folder+'Train_Subset.mat'
Test_CalBased_File = data_folder+'CalBased_Test_Subset.mat'
Test_CalFree_File = data_folder+'CalFree_Test_Subset.mat'
Test_AAMI_File = data_folder+'AAMI_Test_Subset.mat'



# Training model for estimating SBP. Replace 'SBP' with 'DBP' to train model for DBP.
#!!!!!!! need to modify trainer.py about shape gt
BP = 'SBP'
#BP = 'DBP'
Train_Data = Build_Dataset(Train_File, BP)
Test_CalBased_Data = Build_Dataset(Test_CalBased_File, BP)
Test_CalFree_Data = Build_Dataset(Test_CalFree_File, BP)
Test_AAMI_Data = Build_Dataset(Test_AAMI_File, BP)


# %% Start model training

if __name__ == '__main__':
    # Initialize model 
    Seed(6)
    #model = Mtransformerv2.ViT1D_Fusion(
    #   in_channels=1, seq_len=1250, patch_size=10, emb_dim=128, 
    #    depth=6, nhead=8, mlp_ratio=4, num_BP=1, dropout=0.1, num_static_features=3)
    model = Mtransformerv3.ViT1D_FiLM(
        in_channels=1, seq_len=1250, patch_size=10, emb_dim=128, 
        depth=6, nhead=8, mlp_ratio=4, num_BP=1, dropout=0.1)
    #model = Mtransformerv3_noshape.ViT1D_FiLM(
    #    in_channels=1, seq_len=1250, patch_size=10, emb_dim=128, 
    #    depth=6, nhead=8, mlp_ratio=4, num_BP=1, dropout=0.1)
    #model = Mtransformerv4.ViT1D_ConditionalAttn(
    #    in_channels=1, seq_len=1250, patch_size=10, emb_dim=128, 
    #    depth=6, nhead=8, mlp_ratio=4, num_BP=1, dropout=0.1)
    Seed(6)
    
    # Prepare settings to be recorded
    Settings = {'BP_optimizer': 'torch.optim.Adam(model.parameters(), lr=4e-5, betas=(0.9, 0.999), weight_decay=1e-8)',
                'trainer': 'Model_Trainer(model,torch.nn.MSELoss(), BP_optimizer, device, Settings,batch_size=32, num_epochs=100, save_states=True, save_final=True)'
                }
    # Setup training device
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(BP)
    print(device)
    print(torch.cuda.get_device_name(0))
    model.to(device)
    # Instantiate optimizer and model trainer
    BP_optimizer = eval(Settings['BP_optimizer'])
    model_trainer = eval(Settings['trainer'])
    # Set the training set and the two setting set under comparison
    model_trainer.Set_Dataset(Train_Data, {
                              'Test_CalBased': Test_CalBased_Data, 'Test_CalFree': Test_CalFree_Data, 'Test_AAMI': Test_AAMI_Data})
    model_trainer.Train_Model()
    # Find the curves of error metrics in the TensorBoard folder
