import torch
import torch.nn as nn
import torch.utils.data as data
import random
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
from mat73 import loadmat
from Model_Def.Trainerv2_warmup import Model_Trainer
from Model_Def import Mtransformerv2, Mtransformerv3


def lr_lambda(epoch: int):
    # epoch 从 0 开始
    if epoch < warmup_epochs:
        # 线性 warmup: 0 -> 1
        return float(epoch + 1) / float(warmup_epochs)
    # warmup 后保持 1.0 倍
    return 1.0


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
    def __init__(self, Input, Age, BMI, Gender, Label, augment=False, noise_std=0.1):
        self.Input = Input  # (N, 1, seq_len)
        self.Age = Age  # (N,)
        self.BMI = BMI  # (N,)
        self.Gender = Gender  # (N,)
        self.Label = Label  # (N,)
        self.augment = augment  # 新增：是否开启增强
        self.noise_std = noise_std  # 新增：噪声强度

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        # 获取基础数据
        age_val = self.Age[idx]
        bmi_val = self.BMI[idx]

        # === 如果开启增强（只用于 Training），加入随机噪声 ===
        if self.augment:
            # 这里的噪声是基于 z-score 后的数值。
            # std=0.1 意味着波动范围大约是原始数据标准差的 10%
            age_val += np.random.normal(0, self.noise_std)
            bmi_val += np.random.normal(0, self.noise_std)

        # Gender 是 0/1 离散值，通常不加噪声，或者你可以加极小的噪声但一般没必要
        static_feat = np.array([age_val, bmi_val, self.Gender[idx]], dtype=np.float32)

        return (self.Input[idx, :].astype(np.float32), static_feat), self.Label[idx]  # x, y


def z_score_normalize(array):
    mean = np.mean(array)
    std = np.std(array)
    if std == 0:
        return np.zeros_like(array)  # 避免除零
    return (array - mean) / std


def Build_Dataset(Path, Label, show_info=True, augment=False):  # 新增 augment 参数
    Data = loadmat(Path)
    signals = np.expand_dims(Data['Subset']['Signals'][:, 1, :], axis=1)
    age = np.array(Data['Subset']['Age'], dtype=np.float32).flatten()
    bmi = np.array(Data['Subset']['BMI'], dtype=np.float32).flatten()
    gender_raw = Data['Subset']['Gender']
    gender = encode_gender(gender_raw)
    
    label = np.array(Data['Subset'][Label], dtype=np.float32).flatten()

    # 过滤 nan/inf
    N = len(age)
    before_num = N
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        # 将 is_normal 也加入无效值检查（虽然二值数据通常没问题，但为了严谨建议加入）
        features = [age[i], bmi[i], gender[i]]
        if (
            np.any(np.isnan(features)) or np.any(np.isinf(features)) or
            np.any(np.isnan(signals[i])) or np.any(np.isinf(signals[i]))
        ):
            mask[i] = False
            
    # 应用 mask 过滤所有数组
    signals = signals[mask]
    age = age[mask]
    bmi = bmi[mask]
    gender = gender[mask]
    label = label[mask]
    after_num = len(age)

    # === z-score 标准化 ===
    age = z_score_normalize(age)
    bmi = z_score_normalize(bmi)

    if show_info:
        print(f"{Path}: Before removing nan/inf: {before_num}, After: {after_num}")

    return Dataset(signals, age, bmi, gender, label, augment=augment, noise_std=0.1)




data_folder =  "/home/yshen28/PPGdata/"
Train_File = data_folder+'Train_Subset.mat'
Test_CalBased_File = data_folder+'CalBased_Test_Subset.mat'
Test_CalFree_File = data_folder+'CalFree_Test_Subset.mat'
Test_AAMI_File = data_folder+'AAMI_Test_Subset.mat'



# Training model for estimating SBP. Replace 'SBP' with 'DBP' to train model for DBP.
#!!!!!!! need to modify trainer.py about shape gt
#BP = 'SBP'
BP = 'DBP'
Train_Data = Build_Dataset(Train_File, BP, augment=True)

# no noise for test
Test_CalBased_Data = Build_Dataset(Test_CalBased_File, BP, augment=False)
Test_CalFree_Data = Build_Dataset(Test_CalFree_File, BP, augment=False)
Test_AAMI_Data = Build_Dataset(Test_AAMI_File, BP, augment=False)


# %% Start model training

if __name__ == '__main__':
    warmup_epochs = 5
    base_lr = 4e-5
    num_epochs = 100  
    
    

    # Initialize model 
    #Seed(6)
    Seed(6)

    model = Mtransformerv3.ViT1D_FiLM(
        in_channels=1, seq_len=1250, patch_size=10, emb_dim=128, 
        depth=6, nhead=8, mlp_ratio=4, num_BP=1, dropout=0.1)
        
    #Seed(6)
    Seed(42)
    
    # Prepare settings to be recorded
    Settings = {
        'BP_optimizer': 'Adam lr=4e-5 + warmup 5 epochs',
        'trainer': 'Model_Trainer with scheduler'
    }
    
    # Setup training device
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(BP)
    print(device)
    print(torch.cuda.get_device_name(0))
    model.to(device)
    # Instantiate optimizer and model trainer
    BP_optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-8)
    
    scheduler = LambdaLR(BP_optimizer, lr_lambda=lr_lambda)
    
    model_trainer = Model_Trainer(
        model, torch.nn.MSELoss(), BP_optimizer, device, Settings,
        batch_size=32, num_epochs=100, save_states=True, save_final=True,
        scheduler=scheduler
    )
    
    # Set the training set and the two setting set under comparison
    model_trainer.Set_Dataset(Train_Data, {
                              'Test_CalBased': Test_CalBased_Data, 'Test_CalFree': Test_CalFree_Data, 'Test_AAMI': Test_AAMI_Data})
    model_trainer.Train_Model()
    # Find the curves of error metrics in the TensorBoard folder
