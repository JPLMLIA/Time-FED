#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:27:26 2024

@author: marchett
"""

import argparse, sys
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from timefed.utils import utils
from pathlib import Path


from mlky import (
    Config,
    Sect
)

from sklearn.metrics import (
    # Classification metrics
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    # Regression metrics
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)


Logger = logging.getLogger('timefed/model.py')




#define LSTM architecture
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size, target_size, num_layers,
                 dropout =0.1):
        super(LSTMClassifier, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.lstm = torch.nn.LSTM(
            input_size = n_features, 
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True)
        # Define the dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(
            in_features = hidden_size,
            out_features= target_size)

    def forward(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        #print(f'x size: {x.size()}')
        #initialize hidden and cell states with zeros
        h0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size),
                          device = x.device)
        c0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size),
                          device = x.device)
        
        #truncated backpropagation through time with detach
        out, (h, c) = self.lstm(x, (h0, c0))
        #print(f'out size: {out.size()}')
        out = self.dropout(out)
        #-1 is the index of hidden state at last time step
        logits = self.fc(out[:, -1, :])
        #print(f'logits size: {logits.size()}')
        #scores = torch.sigmoid(logits)

        return logits
    
    
    
# make pytorch compartible dataset, takes windowed data 
class TDLDataset(Dataset):
    def __init__(self, windowed_df, cols, means, stds, down_ratio=1., up_ratio=1.):
        self.label_col = 'Label'
        #self.cols = cols
        self.means = means
        self.stds = stds
        
        self.cols = cols
        #self.cols = ['diff_AGC_VOLTAGE', 'diff_CARRIER_SYSTEM_NOISE_TEMP']
        windowed_data = np.array(windowed_df[self.cols].to_dataarray())
        windowed_data = np.transpose(windowed_data, (1,2,0))
        
        num_windows = len(windowed_data)
        
        for i in range(num_windows):
            window_means = np.mean(windowed_data[i, :, :], axis = 0)
            # rolling mean for longer windows?
            # col_std = np.std(windowed_data[i, :, :], axis = 0)
            # col_std[col_std == 0] = 1e-10
            
            windowed_data[i, :, :] += self.means
            windowed_data[i, :, :] /= self.stds
                
            windowed_data[i, :, :] -= window_means
            #windowed_data[i, :, :] /= col_std
        
        labels = np.array(windowed_df['Label'])
        
        if (down_ratio != 1.) | (up_ratio != 1.):
            self.indices = balance_sample(windowed_data, labels, 
                                          down_ratio, up_ratio)
        else:
            self.indices = np.arange(len(windowed_data))
      
        self.data =  windowed_data[self.indices]
        self.labels = labels[self.indices]

                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve the data point at the specified index
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Apply transformations of mean+std norm and to tensor
        sample = transforms.ToTensor()(sample).float()
        
        return sample, label    
 



# upsample and downsample classes
def balance_sample(dataset, labels, downsampling_ratio, upsampling_ratio):
    # Calculate the number of samples to keep for each class
    n_per_class = {}
    class_labels = [0., 1.]
    minority_label = 1.
    for label in class_labels:
        num_samples = sum(1 for lbl in labels if lbl == label)
        if label == minority_label:
            n_per_class[label] = int(num_samples * upsampling_ratio)
        else:
            n_per_class[label] = int(num_samples * downsampling_ratio)
            

    indices = []
    for label in class_labels:
        label_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        if len(label_indices) == 0:
            continue  # Skip if no samples for this class
        if (label == minority_label) & (upsampling_ratio > 1.):
            sampled_indices = torch.randint(len(label_indices), (n_per_class[label],))
        else:
            sampled_indices = torch.randperm(len(label_indices))[:n_per_class[label]]
        indices.extend(label_indices[i] for i in np.sort(sampled_indices))
        
    indices = np.sort(indices) 
    
    return indices



@utils.timeit
def main():
    """
    Builds a model, trains and tests against it, then creates plots for the run.

    Parameters
    ----------
    config : utils.Config
        Config object defining arguments for classify
    """
    # Retrieve this script's configuration section to reduce verbosity of code
    C = Config.model
    Logger.info(f'Getting test and train from: {C.file}')

    data = Sect({
        'train': xr.open_dataset(C.file, group = 'select/train'),
        'test' : xr.open_dataset(C.file, group = 'select/test')
    })
    Logger.info('Done reading train and test.')
    
    cols = ['diff_AGC_VOLTAGE', 'diff_CARRIER_SYSTEM_NOISE_TEMP']
    batch_size = C.model.parameters.batch
    learning_rate = C.model.parameters.rate
    n_epochs = C.model.parameters.epochs

    # Either load an existing model file
    if Path(path := C.model.file).exists() and not C.model.overwrite:
        Logger.info(f'Loading existing model: {path}')
        model = utils.load_pkl(path)
    
    # or create a new one
    else:
        C.model.fit = True

        Logger.info(f'model type: {C.type}')
        if C.type == 'regression':
            print('Regression is not available')
            sys.exit()
            
        window_size = data.train['datetime'].shape[1]
        
        means = np.array(data.train[cols].mean().to_array())
        stds = np.array(data.train[cols].std().to_array())
        
        Logger.info('Creating DL-ready train and validation data.')
        train_data = TDLDataset(data.train, cols, means, stds, down_ratio = 0.25,
                                     up_ratio = 1)  
        train_loader = DataLoader(train_data, batch_size = batch_size, 
                                  shuffle = True)
        
        val_data = TDLDataset(data.test, cols, means, stds, down_ratio = 0.1, 
                              up_ratio = 0.25)
        val_loader = DataLoader(dataset = val_data, batch_size = batch_size, 
                                shuffle = False)
    
        Logger.info('Starting model training.')
        ###### start modeling and training
        model = LSTMClassifier(n_features= len(cols), 
                                hidden_size=window_size // 2,
                                dropout = 0.1,
                                target_size=1,
                                num_layers=2)
        
        
        loss_function = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        history = {'loss': [], 'val_loss': []}
        for epoch in range(n_epochs):
            losses = []
            train_pred_labels = []
            true_labels = []
    
            model.train()
            for i, batch in enumerate(train_loader):
                
                #model.train()
                inputs, labels = batch
    
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = loss_function(outputs, labels.reshape(-1,1).float())
                losses.append(loss.item())
                
                # Backward pass and optimization
                optimizer.zero_grad()  # Zero gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update parameters
                
                train_pred_labels.append(np.hstack(torch.sigmoid(outputs) > 0.5) * 1)
                true_labels.append(labels)
    
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels.reshape(-1,1).float())
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
    
    
            avg_loss = np.mean(losses)
            history['loss'].append(avg_loss)
            history['val_loss'].append(val_loss)
            acc = accuracy_score(np.hstack(true_labels), 
                                         np.hstack(train_pred_labels))
            cm = confusion_matrix(np.hstack(true_labels),
                                          np.hstack(train_pred_labels))
            tpr = cm[1,1] / (cm[1,1] + cm[1,0])
            
            train_message = f'Loss = {avg_loss:0.2f}, Acc = {acc:0.2f}, TPR = {tpr:0.2f}'
            val_message = f'Val Loss: {val_loss:.2f}'
            Logger.info(f'Epoch {epoch+1} / {n_epochs}: {train_message}, {val_message}' )
            #print(f'Epoch {epoch+1} / {n_epochs}: {train_message}, {val_message}' )
        
        
        # TO ADD:
        # pr = metrics.precision_score(np.hstack(true_labels), np.hstack(train_pred_labels))
        # rec = metrics.recall_score(np.hstack(true_labels), np.hstack(train_pred_labels))   
        # plt.figure()
        # plt.plot(history['loss'], '.-', ms = 3, label = 'training loss')
        # plt.plot(history['val_loss'], '.-', ms = 3, label = 'valid loss')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.text(x = 0, y = 10, s= cm.astype(str))
        # plt.grid(ls=':', alpha = 0.5)
        # plt.legend()
        # plt.title(f'acc = {acc:0.2f}, pr = {pr:0.2f}, rec = {rec:0.2f}\n n windows = {len(train_data.data)}, n pos = {train_data.labels.sum()}')
        # plt.savefig(f'{output_dir}/loss_train.png')  
        # plt.close()
        
        
        # Save the newly trained model
        if C.output.model:
            exists = Path(path := C.model.file).exists()
            if not exists or (exists and C.model.overwrite):
                utils.save_pkl(path, model)
                
                    
        ######## test the model
        Logger.info('Testing model.')
        test_data = TDLDataset(data.test, cols, means, stds, down_ratio = 1.,
                                     up_ratio = 1.)
        test_loader = DataLoader(test_data, batch_size = batch_size,
                                 shuffle = False)
        model.eval()
        pred_labels = []
        true_labels = []
        proba = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
    
                # Forward pass, get scores
                outputs = model(inputs[:,0,:,:])
                proba.append(np.hstack(torch.sigmoid(outputs)))
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                pred_labels.append(np.hstack(preds))
                true_labels.append(np.hstack(labels))  
        pred_labels = np.hstack(pred_labels)
        true_labels = np.hstack(true_labels)
        proba = np.hstack(proba)
        
        Logger.info('Saving results.')
        predicts = pd.DataFrame(np.vstack([pred_labels, true_labels, proba]).T, 
                          columns = ['pred', 'actual', 'proba'])
        predicts['time'] = data.test['datetime'].values[:, 0:1]
        col_keys = ['GROUND_ANTENNA_ID', 'SCHEDULE_ITEM_ID','DCC_CHANNEL']
        for k in col_keys:
            predicts[k] = data.test[k].values[:,0]
            
        predicts = predicts.set_index(['time'])
        #ds = xr.Dataset.from_dataframe(ds)
        #ds.to_netcdf(f'{C.output.predicts}')
        
        if C.output.predicts:
            predicts.to_hdf(C.output.predicts, 'predicts/test')
        
        if C.evaluate:
            Logger.info('Evaluating results.')
            from timefed.core.evaluate import evaluate_deep
            evaluate_deep(C.output.predicts, Config.name)
        # not suppoerted
        # if C.train_scores:
        #     scores['train'], predicts = score(model, data.train, 'train')
        
        #     if C.output.predicts:
        #         predicts.to_hdf(C.output.predicts, 'predicts/train')
        
        # if C.output.scores:
        #     utils.save_pkl(C.output.scores, scores)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--config',   type     = str,
                                            required = True,
                                            metavar  = '/path/to/Config.model.yaml',
                                            help     = 'Path to a Config.model.yaml file'
    )
    parser.add_argument('-s', '--section',  type     = str,
                                            default  = 'model',
                                            help     = 'Section of the config to use'
    )

    args = parser.parse_args()

    try:
        utils.init(args)

        code = main()

        Logger.info('Finished successfully')
    except Exception as e:
        Logger.exception('Failed to complete')
        
    
    
    
    
    

       
            





    
 