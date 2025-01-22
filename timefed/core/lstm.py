import argparse
import logging
import sys
from pathlib import Path

import numpy  as np
import pandas as pd
import xarray as xr
from mlky import (
    Config,
    Sect
)
from sklearn.metrics import (
    # Classification metrics
    accuracy_score,
    confusion_matrix,
)

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from timefed.utils import evaluate
from timefed.utils import utils

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

Logger = logging.getLogger('timefed/nn')


class LSTMClassifier(torch.nn.Module):
    def __init__(self, n_features, hidden_size, target_size, num_layers=1, dropout=0.1):
        """
        Long-Short Term Memory neural network model

        Parameters
        ----------
        n_features : int
            The number of expected features in the input `x`
        hidden_size : int
            The number of features in the hidden state `h`
        target_size : int
            Size of the target
        num_layers : int, default=1
            Number of recurrent layers. E.g., setting ``num_layers=2`` would mean
            stacking two LSTMs together to form a `stacked LSTM`, with the second LSTM
            taking in outputs of the first LSTM and computing the final results
        dropout : float, default=0.1
            TODO

        Returns
        -------

        """
        super(LSTMClassifier, self).__init__()

        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        self.target_size = target_size

        self.lstm = torch.nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True
        )

        # Define the dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(
            in_features  = hidden_size,
            out_features = target_size
        )

    def forward(self, x):
        """
        TODO

        Parameters
        ----------
        x : numpy.ndarray
            TODO

        Returns
        -------
        logits : TODO
            TODO
        """
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)

        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(
            (self.num_layers, x.size(0), self.hidden_size),
            device = x.device
        )
        c0 = torch.zeros(
            (self.num_layers, x.size(0), self.hidden_size),
            device = x.device
        )

        # Truncated backpropagation through time with detach
        out, (h, c) = self.lstm(x, (h0, c0))
        out = self.dropout(out)

        # -1 is the index of hidden state at last time step
        logits = self.fc(out[:, -1, :])

        return logits


class TDLDataset(Dataset):
    def __init__(self, ds, target, columns, means, stds, down_ratio=1., up_ratio=1.):
        """
        PyTorch-compatible Dataset of TimeFED xarray dataset objects

        Parameters
        ----------
        ds : xr.Dataset
            TimeFED dataset output from the select module. Must be 2D with dimensions
            `windowID` and `index`
        target : str
            Name of the labels variable
        columns : list[str]
            List of feature columns
        means : numpy.ndarray
            Means of the training dataset
        stds : numpy.ndarray
            Standard deviations of the training dataset
        down_ratio : float, default=1.0
            Ratio for sampling the negative class. Less than 1 is downsampling, greater
            than one is upsampling. Upsampling will create duplicates.
        up_ratio : float, default=1.0
            Ratio for sampling the positive class. Less than 1 is downsampling, greater
            than one is upsampling. Upsampling will create duplicates.
        """
        self.target  = target
        self.columns = columns
        self.n_features = len(columns)

        self.means = means
        self.stds  = stds

        # Track the timestamps for this dataset
        self.time = ds['datetime'].isel(index=0)

        data = ds[columns].to_array().transpose('windowID', 'index', 'variable')
        data = (data + means) / stds - data.mean('index')

        labels = ds[target]
        self.indices = data.windowID.data
        if not (down_ratio == up_ratio == 1):
            self.indices = balance_classes(labels, down_ratio, up_ratio)

            data   = data.isel(windowID=self.indices)
            labels = labels.isel(windowID=self.indices)

        self.data   = torch.FloatTensor(data.data)
        self.labels = labels.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def balance_classes(labels, negRatio=1.0, posRatio=1.0):
    """
    Balances positive/negative classes with given ratios

    Parameters
    ----------
    labels : xarray.DataArray
        Labels data
    negRatio : float, default=1.0
        Ratio for sampling the negative class. Less than 1 is downsampling, greater
        than one is upsampling. Upsampling will create duplicates.
    posRatio : float, default=1.0
        Ratio for sampling the positive class. Less than 1 is downsampling, greater
        than one is upsampling. Upsampling will create duplicates.

    Returns
    -------
    indices : np.ndarray
        Selected indices
    """
    # Add indices to the dimension
    labels['windowID'] = range(labels.windowID.size)

    # Groupby the class label
    classes = labels.groupby(labels)

    # Calculate number of samples for each class
    counts = classes.count()
    neg = int(counts[0] * negRatio)
    pos = int(counts[1] * posRatio)

    # Now select indices randomly
    indices = np.hstack([
        np.random.choice(classes[0].windowID, neg),
        np.random.choice(classes[1].windowID, pos)
    ])
    indices.sort()

    return indices


@utils.timeit
def train_LSTM(train, val, hidden_size, learning_rate, n_epochs):
    """
    Trains and validates an LSTM model

    Parameters
    ----------
    train : DataLoader
        Training dataset
    val : DataLoader
        Validation dataset

    Returns
    -------
    model : LSTMClassifier
        Trained LSTM model
    """
    Logger.info('Training LSTM model')
    model = LSTMClassifier(
        n_features  = train.dataset.n_features,
        hidden_size = hidden_size,
        dropout     = 0.1,
        target_size = 1,
        num_layers  = 2
    )

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {'loss': [], 'val_loss': []}
    for epoch in range(n_epochs):
        losses = []
        train_pred_labels = []
        true_labels = []

        model.train()
        for i, (inputs, labels) in enumerate(train):
            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_function(outputs, labels.reshape(-1,1).float())
            losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad() # Zero gradients
            loss.backward()       # Compute gradients
            optimizer.step()      # Update parameters

            train_pred_labels.append(np.hstack(torch.sigmoid(outputs) > 0.5) * 1)
            true_labels.append(labels)


        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val:
                outputs = model(inputs)
                loss = loss_function(outputs, labels.reshape(-1,1).float())
                val_loss += loss.item()

        val_loss /= len(val)


        avg_loss = np.mean(losses)
        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss)

        true = np.hstack(true_labels)
        pred = np.hstack(train_pred_labels)

        acc = accuracy_score(true, pred)
        cm  = confusion_matrix(true, pred)
        tpr = cm[1,1] / (cm[1,1] + cm[1,0])

        Logger.info(f'Epoch {epoch+1} / {n_epochs}: Loss={avg_loss:0.2f}, Acc={acc:0.2f}, TPR={tpr:0.2f}, ValLoss={val_loss:0.2f}')

    return model


def test_LSTM(model, test, output, threshold=0.5):
    """
    Tests a trained LSTM model against a test dataset and saves the results to an HDF5
    file

    Parameters
    ----------
    model : ...
        LSTM model
    test : ...
        Test dataset
    output : str
        Output HDF5 path. Will be saved under key `predicts/test`
    threshold : float, default=0.5
        Threshold for classifying a prediction probability to be positive
    """
    data = {
        'truth': [],
        'predict': [],
        'probability': []
    }

    model.eval()
    with torch.no_grad():
        for inputs, labels in test:
            # Forward pass, get scores
            outp = model(inputs)
            prob = torch.sigmoid(outp)
            pred = (prob >= threshold).int()

            data['truth'] += labels.tolist()
            data['predict'] += pred.flatten().tolist()
            data['probability'] += prob.flatten().tolist()

    df = pd.DataFrame(data, index=test.dataset.time)
    df.to_hdf(output, 'predicts/test')


@utils.timeit
def main():
    """
    Builds a model, trains and tests against it, then creates plots for the run.
    """
    # Retrieve this script's configuration section to reduce verbosity of code
    C = Config.model
    params = C.model.parameters

    Logger.info('Loading test and train')
    Logger.debug(f'File: {C.file}')

    # data = Sect(
    #     train = xr.open_dataset(C.file, group='select/train'),
    #     test  = xr.open_dataset(C.file, group='select/test')
    # )
    data = Sect(
        train = xr.open_dataset('train.nc'),
        test  = xr.open_dataset('test.nc')
    )
    means = np.array(data.train[params.cols].mean().to_array())
    stds  = np.array(data.train[params.cols].std().to_array())

    # Either load an existing model file
    if Path(path := C.model.file).exists() and not C.model.overwrite:
        Logger.info(f'Loading existing model: {path}')
        model = utils.load_pkl(path)

    # or create a new one
    else:
        Logger.debug(f'Model type: {C.type}')
        if C.type == 'regression':
            Logger.error('Regression is not supported for deep models at this time')
            return

        Logger.info('Creating DL-ready train and validation datasets')
        model = train_LSTM(
            # Training set
            train = DataLoader(
                TDLDataset(data.train, C.target, params.cols, means, stds,
                    down_ratio = 0.25,
                    up_ratio   = 1.00
                ),
                batch_size = params.batch,
                shuffle    = True
            ),
            # Validation set
            val = DataLoader(
                TDLDataset(data.test, C.target, params.cols, means, stds,
                    down_ratio = 0.10,
                    up_ratio   = 0.25
                ),
                batch_size = params.batch,
                shuffle    = False
            ),
            hidden_size   = data.train.index.size // 2,
            learning_rate = params.rate,
            n_epochs = params.epochs,
        )

        # Save the newly trained model
        if C.output.model:
            exists = Path(path := C.model.file).exists()
            if not exists or (exists and C.model.overwrite):
                utils.save_pkl(path, model)

    if file := C.output.predicts:
        Logger.info('Testing model')

        test_LSTM(
            model = model,
            test = DataLoader(
                TDLDataset(data.test, 'Label', params.cols, means, stds,
                    down_ratio = 1.00,
                    up_ratio   = 1.00
                ),
                batch_size = params.batch,
                shuffle    = False
            ),
            output = file,
            threshold = 0.5
        )

        if C.evaluate:
            Logger.info('Evaluating results')
            evaluate.lstm()


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
