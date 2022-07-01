### PyTourch_util:
# It contains functions useful for building models with PyTorch
# ------------------------------------------------------------------------------
### LIBRARIES/PACKAGES ---------------------------------------------------------
import sys, os
import numpy as np
import torch
import torch.nn as nn


# Default weights of various layer types of PyTorch

# My own default weights for PyTorch layers
# in __init__, do: self.apply(L_default_weights)
def L_default_weights(m):
  # Convolution layer
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data, nonlinearity = 'relu')
      if m.bias is not None: nn.init.constant_(m.bias.data, 0)

  # Linear layer
  elif isinstance(m, nn.Linear):
      nn.init.xavier_normal_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

  # RNN layer
  elif isinstance(m, nn.RNN):
    for name, param in m._parameters.items():
         if 'bias' in name:
             nn.init.constant_(param, 0.0) # bias of both layer

         elif 'weight_ih' in name: # weight of input-hidden layer
             nn.init.xavier_normal_(param)

         elif 'weight_hh' in name: # weight of hidden-hidden layer
             nn.init.orthogonal_(param)

  # LSTM layer
  elif isinstance(m, nn.LSTM):
    for name, param in m._parameters.items():
         if 'bias' in name:
             nn.init.constant_(param, 0.0) # bias of both layer

         elif 'weight_ih' in name: # weight of input-hidden layer
             nn.init.xavier_normal_(param)

         elif 'weight_hh' in name: # weight of hidden-hidden layer
             nn.init.orthogonal_(param)

         elif 'weight_hr' in name: # weight of projection layer
             nn.init.xavier_normal_(param)

  # GRU layer
  elif isinstance(m, nn.GRU):
    for name, param in m._parameters.items():
         if 'bias' in name:
             nn.init.constant_(param, 0.0) # bias of both layer

         elif 'weight_ih' in name: # weight of input-hidden layer
             nn.init.xavier_normal_(param)

         elif 'weight_hh' in name: # weight of hidden-hidden layer
             nn.init.orthogonal_(param)

  # Batch normalization layer
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)

### TRAINER --------------------------------------------------------------------
# Trainer function for all PyTorch models!
# model = nn.Module with specific layers
# s  = a dictionary {n_epochs, device, optimizer, criterion, loaders}
# (tracer) = accuracy and loss
def Trainer(model, s, report = 100):
    """
    Trainer function for all PyTorch models!
    model = nn.Module with specific layers
    s  = a dictionary {n_epochs, device, optimizer, criterion, loaders}
    (tracer) = accuracy and loss
    """
    # Define all setting dictionary as variables
    # locals().update(s)
    n_epochs = s['n_epochs']
    device = s['device']
    loaders = s['loaders']
    optimizer = s['optimizer']
    criterion = s['criterion']
    tracer = {'loss_trn': [], 'loss_tst': [], 'acc_trn' : [], 'acc_tst' : []}
    print('---Training in progress!---')

    # Loop over n_epoch times
    for epoch in range(n_epochs):

        for phase in ['trn', 'tst']:
            # Change model mode depending on the data set
            model.train()
            if phase == 'tst':model.eval()
            if (epoch == 0) and (phase == 'trn'):continue

            # Define evaluation variables
            loss_sum, acc_sum = 0.0, 0.0

            for i , (inputs, targets) in (enumerate(loaders[phase])):
                # CPU/GPU stuff (don't worry)
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()

                # Initialize gradient of optimizer
                optimizer.zero_grad()

                # Do training only in train
                with torch.set_grad_enabled(phase == 'trn'):
                    # Forward path
                    outputs = model(inputs)  # assumes first return is output
                    if isinstance(outputs, tuple):outputs = outputs[0]

                    # Calculate loss: difference between output and label
                    loss = criterion(outputs, targets)

                    # Backward path (Backpropagation!)
                    if phase == 'trn':
                        loss.backward()
                        optimizer.step()

                    # Keep track of the progress of learning
                    # This bit depends on loss function...(regression vs categorization)
                    loss_sum += loss

                    if type(criterion).__name__ == 'CrossEntropyLoss':
                       _, preds = torch.max(outputs, 1) # prediction = max softmax output
                       acc_sum += torch.sum(preds == targets.data)

                    if type(criterion).__name__ == 'MSELoss':
                       acc_sum += torch.sum(torch.abs(targets - outputs))

            # Display learning progress
            loss_sum = loss_sum / len(loaders[phase].dataset)
            acc_sum = acc_sum.double() / len(loaders[phase].dataset)
            tracer['loss_' + phase].append(loss_sum) # cumulative loss/ batch size
            tracer['acc_' + phase].append(acc_sum) # cumulative accuracy/ batch size
            if epoch > 0 and epoch % report == 0:
                print('Phase:{}, Epoch:{}, Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch, loss_sum, acc_sum))


    print('---Done!---')
    return tracer
