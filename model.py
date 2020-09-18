"""
contains model and training function
"""
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM module
    Args:
        vocab_size: size of vocabulary (len of word2index)
        embedding_dim: size of embedding dimension
        hidden_size: no of features in hidden state
        n_layers: no of recurrent  layers
        output_size: number of output classes to be returned
    """
    def __init__(
            self,
            vocab_size,
            embedding_dim=100,
            hidden_size=124,
            output_size=4,
            n_layers=124,
    ):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        embed = self.embed(inputs)
        output, _ = self.lstm(embed)
        fc1 = self.fc1(output[:, :, -1]).squeeze()
        return fc1


def train(epochs, train_loader, valid_loader, model, optimizer, criterion, save_path, device, stopping=None, batch=20):
    """
    Trains LSTM model

    Args:
        epochs: no of iterations to ru
        train_loader: training data loader
        valid_loader: validation data loader
        model: LSTM model
        optimizer: optimizer (Adam, SGD,..)
        criterion: loss
        save_path: location to save model
        device: type of device to be used while training
        batch: no of batches used to build dataloader
        stopping: Default: None,
                  if 'acc' saves model when validation accuracy increases on epoch
                  if 'loss' saves model when validation loss decreases  on epoch
    Returns:

    """
    max_validation_acc = 1
    min_validation_loss = np.Inf
    history = {'training_loss': [],
               'validation_loss': [],
               'training_acc': [],
               'validation_acc': []
               }
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, leave=True)
        training_loss = 0
        validation_loss = 0
        training_acc = 0
        validation_acc = 0

        # Training
        model.train()
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            _, preds = torch.max(output, dim=1)
            training_acc += (preds == targets).sum().item()
            loss = criterion(output, targets)
            loss.backward()
            training_loss = loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            progress_bar.set_description('Training loss: {}'.format(training_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                _, preds = torch.max(output, dim=1)
                validation_acc += (preds == targets).sum().item()
                loss = criterion(output, targets)
                validation_loss = loss.item()

        training_acc = 100 * training_acc / (len(train_loader)*batch)
        validation_acc = 100 * validation_acc / (len(valid_loader)*batch)

        history['validation_loss'].append(validation_loss)
        history['training_loss'].append(training_loss)
        history['validation_acc'].append(validation_acc)
        history['training_acc'].append(training_acc)
        print(
            'Epoch: {}\tTraining Loss: {:.6f}\tValidation Loss: {:.6f}\n '
            'Training Accuracy: {:.6f}\tValidation Accuracy: {:.6f}'.format(
                epoch, training_loss, validation_loss, training_acc, validation_acc))
        if stopping == 'acc':
            if validation_acc >= max_validation_acc:
                print('Validation accuracy increased...........')
                print('saving model')
                torch.save(model, save_path)
                max_validation_acc = validation_acc

        if stopping == 'loss':
            if validation_loss <= min_validation_loss:
                print('Validation loss decreased...........')
                print('saving model')
                torch.save(model, save_path)
                min_validation_loss = validation_loss
    return model, history
