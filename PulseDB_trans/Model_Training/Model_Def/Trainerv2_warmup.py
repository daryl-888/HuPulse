from datetime import datetime
import time
import torch
import torch.utils.data as data
import os
import torch.nn as nn
import numpy as np
import progressbar as PB
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter as SW
from io import StringIO
import sys

def check_tensor_valid(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN")
    if torch.isinf(tensor).any():
        print(f"{name} contains Inf")

# R-Squared
def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Mean error
#def ME(y_true, y_pred):
#    return np.mean(y_true-y_pred)
    
def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Standard deviation of error
def SD(y_true, y_pred):
    return np.std(y_true-y_pred)


widgets = [
    PB.Bar(),
    PB.Counter(),
    ' ',
    PB.Percentage(),
    ' ',
    #PB.DynamicMessage('Batch_BP_Loss'),
    PB.DynamicMessage('BP_Loss'),
    PB.DynamicMessage('Shape_Loss'),

    ' ',
    PB.ETA()
]


class Model_Trainer:
    def __init__(self, model, criterion_BP, optimizer_BP, device, settings_yml, batch_size=32, num_epochs=100, save_states=False, save_final=False, scheduler=None):

        self.Model_Running = model.to(device)
        self.Model_BestTest = []
        self.BP_Loss_Fun = criterion_BP
        self.Optimizer_BP = optimizer_BP
        self.Num_Epoch = num_epochs
        self.Train_Batchsize = batch_size
        self.Device = device
        self.Save_States = save_states
        self.Save_Final = save_final
        self.YMLSettings = settings_yml
        self.Optimizer_BP = optimizer_BP
        self.Scheduler = scheduler

    def Model_Info(self):
        model = self.Model_Running
        print('-' * 10)
        print('Model Structure:')
        print(model)
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable parameters: {}'.format(num))
        print('Settings')
        for item, setting in self.YMLSettings.items():
            print(item, ':', setting)
        print('-' * 10)

    def Set_Dataset(self, train_set, test_set=[]):
        self.Train_Set = train_set
        self.Test_Set_List = test_set

    def Train_Model(self):
        # ???????? __init__ ?
        self.Shape_Loss_Fun = nn.CrossEntropyLoss()
        #self.lambda_shape = 0.5 # ????????????????
        self.log_sigma_bp = nn.Parameter(torch.zeros(1, device=self.Device))
        self.log_sigma_shape = nn.Parameter(torch.zeros(1, device=self.Device))

        TimeID = datetime.now().strftime('%Y_%m%d_%H%M%S')
        ModelID = TimeID[-6:]
        
        Start_Epoch = 1
        batchcounter = 1
        batchrecordcounter = 1
        
        # --- ??:?????????????? ---
        best_metrics = {}
        Test_Names = list(self.Test_Set_List.keys())
        for name in Test_Names:
            best_metrics[name] = {'MAE': float('inf'), 'R2': 0, 'SD': 0, 'Epoch': 0}
        # ---------------------------------------

  
        # Print model info to command line
        Writer = SW(os.path.join('TensorBoard', TimeID))
        print('ModelID: '+ModelID)
        self.Model_Info()
        # Print model info to string so it can be documented in TensorBoard
        ##################################################################################
        save_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        print('ModelID: '+ModelID)
        self.Model_Info()
        sys.stdout = save_stdout
        Writer.add_text('Model', result.getvalue().replace('\n', '     \n'))
        ##################################################################################

        # Set up data loaders for the training and testing sets
        Train = data.DataLoader(
            self.Train_Set, self.Train_Batchsize, shuffle=True, drop_last=False)
        
        Test_Names = []
        Test_List = []
        for name, testdata in self.Test_Set_List.items():
            Test_Names.append(name)
            Test_List.append(data.DataLoader(testdata, batch_size=128))

        Start_Time = time.time()
        # Set up keybaord interrupt, so when training process is interrupted, the model can still be save to files
        Interrupt = False



        Train_Batch = self.Train_Batch

        for Epoch in range(Start_Epoch, Start_Epoch+self.Num_Epoch):
            try:
                print('Epoch {}/{}'.format(Epoch, Start_Epoch+self.Num_Epoch-1))
                print('-' * 10)

                # Training Phase
                Epoch_BP_Train_Loss = []
                k = 0
                Epoch_BP_Preds = []
                Epoch_BP_Labels = []
                with PB.ProgressBar(widgets=widgets, max_value=len(Train)) as bar:
                    for (inputs, static_feat), BP_labels in Train:
                        # ??
                        # ?? Train_Batch ???? bp_pred, shape_pred ??????? loss
                        inputs = inputs.to(self.Device)
                        static_feat = static_feat.to(self.Device)
                        BP_labels = BP_labels.to(self.Device)
                        bp_pred, shape_pred = self.Model_Running(inputs, static_feat)  # ??????????
                        # ?? BP ????
                        bp_loss = self.BP_Loss_Fun(bp_pred, BP_labels)
                        # ?? BP_labels ?????????:>140?1,<=140?0
                        shape_labels = (BP_labels > 140).long()
                        # ????????
                        shape_loss = self.Shape_Loss_Fun(shape_pred, shape_labels)
                        # ???
                        #total_loss = bp_loss + self.lambda_shape * shape_loss
                        total_loss = torch.exp(-self.log_sigma_bp)  * bp_loss   + self.log_sigma_bp + torch.exp(-self.log_sigma_shape) * shape_loss + self.log_sigma_shape

                        # ?????
                        self.Optimizer_BP.zero_grad()
                        total_loss.backward()
                        self.Optimizer_BP.step()

                        # ??
                        Epoch_BP_Labels.append(BP_labels.cpu().numpy())
                        Epoch_BP_Preds.append(bp_pred.detach().cpu().numpy())
                        bar.update(k, BP_Loss=bp_loss.item(), Shape_Loss=shape_loss.item())
                        k += 1
                        batchcounter += 1
                        if not batchcounter % 100:
                            Writer.add_scalar('Batch_BP_Loss', bp_loss.item(), batchrecordcounter)
                            Writer.add_scalar('Batch_Shape_Loss', shape_loss.item(), batchrecordcounter)
                            batchrecordcounter += 1
                    
                    # At the end of each epoch, calculate the error metrics on the training set
                    Epoch_BP_Labels = np.concatenate(Epoch_BP_Labels, axis=0)
                    Epoch_BP_Preds = np.concatenate(Epoch_BP_Preds, axis=0)
                    Epoch_Train_R2 = R2(Epoch_BP_Labels, Epoch_BP_Preds)
                    Epoch_Train_MAE = MAE(Epoch_BP_Labels, Epoch_BP_Preds)
                    Epoch_Train_SD = SD(Epoch_BP_Labels, Epoch_BP_Preds)

                    # Calculate the training loss
                    Epoch_BP_Train_Loss = self.BP_Loss_Fun(torch.from_numpy(
                        Epoch_BP_Labels), torch.from_numpy(Epoch_BP_Preds))
                    
                    # Print a summary of training error metrics
                    print('Epoch BP Training Loss: {:e} mae:{} R2: {}'.format(
                        Epoch_BP_Train_Loss, Epoch_Train_MAE, Epoch_Train_R2))
                    
                    # Save a checkpoint
                    if self.Save_States:
                        self.Save_Checkpoint(
                            ModelID, TimeID, Epoch, batchcounter, batchrecordcounter, savemodel=False)

                # Write training results to TensorBoard
                Writer_Loss_Dict = {'Train_BP': Epoch_BP_Train_Loss}
                Writer_R2_Dict = {'Train': Epoch_Train_R2}
                Writer_MAE_Dict = {'Train': Epoch_Train_MAE}
                Writer_SD_Dict = {'Train': Epoch_Train_SD}

                # Testing Phase
                
                # Rrun testing with the current model on each of the testing sets
                for name, Test in zip(Test_Names, Test_List):
                    Test_Name = name
                    Epoch_Test_Loss = []
                    Epoch_Preds = []
                    Epoch_Labels = []
                    # Accumulate predictions batch by batch
                    for (inputs, static_feat), labels in Test:
                        Loss_Per_Batch, Outputs = self.Test_Batch(
                            inputs, static_feat, labels)
                        Epoch_Test_Loss.append(Loss_Per_Batch)
                        Epoch_Labels.append(labels.cpu().detach().numpy())
                        Epoch_Preds.append(Outputs.cpu().detach().numpy())
                    # Calculate the error metrics
                    Epoch_Labels = np.concatenate(Epoch_Labels, axis=0)
                    Epoch_Preds = np.concatenate(Epoch_Preds, axis=0)

                    Epoch_Test_Loss = self.BP_Loss_Fun(torch.from_numpy(
                        Epoch_Labels), torch.from_numpy(Epoch_Preds))
                    Epoch_Test_R2 = R2(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_MAE = MAE(Epoch_Labels, Epoch_Preds)
                    Epoch_Test_SD = SD(Epoch_Labels, Epoch_Preds)
                    # Print a summary
                    print(
                        'Epoch '+Test_Name + ' Loss: {:e} MAE:{} R2: {}'.format(Epoch_Test_Loss, Epoch_Test_MAE, Epoch_Test_R2))
                        
                    # --- ??:????????????? ---
                    if Epoch_Test_MAE < best_metrics[name]['MAE']:
                        best_metrics[name]['MAE'] = Epoch_Test_MAE
                        best_metrics[name]['R2'] = Epoch_Test_R2
                        best_metrics[name]['SD'] = Epoch_Test_SD
                        best_metrics[name]['Epoch'] = Epoch
                    # ---------------------------------------    
                    
                    
                    # Write to TensorBoard
                    Writer_Loss_Dict.update({Test_Name: Epoch_Test_Loss})
                    Writer_R2_Dict.update({Test_Name: Epoch_Test_R2})
                    Writer_MAE_Dict.update({Test_Name: Epoch_Test_MAE})
                    Writer_SD_Dict.update({Test_Name: Epoch_Test_SD})

                ################################################################################################
                Writer.add_scalars('Loss', Writer_Loss_Dict, Epoch)
                Writer.add_scalars('R2', Writer_R2_Dict, Epoch)
                Writer.add_scalars('MAE', Writer_MAE_Dict, Epoch)
                Writer.add_scalars('SD', Writer_SD_Dict, Epoch)
                ################################################################################################
                # ---- LR scheduler step (epoch-level) ----
                if hasattr(self, "Scheduler") and (self.Scheduler is not None):
                    self.Scheduler.step()
                    # ??:??/????lr,????warmup??
                    current_lr = self.Optimizer_BP.param_groups[0]["lr"]
                    Writer.add_scalar("LR", current_lr, Epoch)
                    print(f"LR after scheduler.step(): {current_lr:.6e}")
                # ----------------------------------------


            # If the training is manually stopped by keyboard interruption
            except KeyboardInterrupt:
                print('Earlystopped by interrupt at epoch {:d}'.format(Epoch))
                Interrupt = True
                break
        Writer.close()
        
        # --- ??:????????????????? ---
        print("\n" + "="*40)
        print("SUMMARY: Best Performance (Minimum MAE)")
        for name in Test_Names:
            res = best_metrics[name]
            print(f"Dataset: {name}")
            print(f"  Best Epoch: {res['Epoch']}")
            print(f"  MAE: {res['MAE']:.4f}")
            print(f"  R2:  {res['R2']:.4f}")
            print(f"  SD:  {res['SD']:.4f}")
        print("="*40 + "\n")
        # ---------------------------------------------
        
        
        time_elapsed = time.time() - Start_Time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # Save the model
        self.Save_Checkpoint(ModelID, TimeID, Epoch,
                             batchcounter, batchrecordcounter, savemodel=True)
        if Interrupt:
            raise KeyboardInterrupt

    # Forward propagation of a batch in training mode
    def Train_Batch(self, inputs, BP_labels):
        
        self.Model_Running.train()
        #inputs = inputs.float().to(self.Device)
        ppg, static_feat = inputs
        ppg = ppg.float().to(self.Device)
        static_feat = static_feat.float().to(self.Device) 
        BP_labels = BP_labels.float().to(self.Device)

        self.Model_Running.zero_grad()
        BP_outputs = self.Model_Running(ppg, static_feat)
        # BP_outputs = self.Model_Running(ppg)
        check_tensor_valid(BP_outputs, "BP_outputs")
        #print(f'BP_outputs.shape:{BP_outputs.shape}')
        #print(f'BP_labels.shape:{BP_labels.shape}')
        BP_loss = self.BP_Loss_Fun(BP_outputs, BP_labels)
        BP_loss_report = BP_loss.item()
        BP_loss.backward()
        self.Optimizer_BP.step()

        return BP_loss_report, BP_outputs
    # Forward propagation of a batch in testing mode (inference)
    def Test_Batch(self, inputs, static_feat, labels):

        self.Model_Running.eval()
        ppg = inputs.float().to(self.Device)
        static_feat = static_feat.float().to(self.Device)
        labels = labels.float().to(self.Device)

        with torch.no_grad():
            # When testing, display only the reconstruction loss
            BP_outputs, shape_pred = self.Model_Running(ppg, static_feat)
            # BP_outputs = self.Model_Running(ppg)
            loss = self.BP_Loss_Fun(BP_outputs, labels)
        return loss.item(), BP_outputs
    # Save a checkpoint
    def Save_Checkpoint(self, modelID, timeID, epoch, batchcounter, batchrecordcounter, savemodel=False):
        # Save a dict for every epoch
        foldername = modelID
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
        torch.save({'model_id': modelID,
                    'time_id': timeID,
                    'model_state_dict': self.Model_Running.state_dict(),
                    'optimizer_state_dict': self.Optimizer_BP.state_dict(),
                    'epoch': epoch,
                    'batchcounter': batchcounter,
                    'batchrecordcounter': batchrecordcounter,
                    }, os.path.join(foldername, 'checkpoint_epoch_{}.pth'.format(epoch)))
        if savemodel:
            torch.save(self.Model_Running, os.path.join(
                foldername, 'trained_model.pth'))