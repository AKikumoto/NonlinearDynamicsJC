### PyTourch_util:
# It contains functions useful for building models with PyTorch
# ------------------------------------------------------------------------------
### LIBRARIES/PACKAGES ---------------------------------------------------------
import sys, os
import numpy as np
import einops
import torch
import torch.nn as nn
from scipy.spatial.distance import euclidean
from torch.autograd import Variable


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

### LAYERS ---------------------------------------------------------------------


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

### LINEARIZATION --------------------------------------------------------------
class LA_fixedpoints:
    def __init__(self, model, device, layerN = 'rnn', q_tor = 1e-05, iter_noise = .1, iter_fp = 10000, lr_fp = .1, lr_decay_tor = 1e-4 ,lr_decay_epoch = 3000):
        # model
        self.model = model
        self.device = device
        self.layerN = 'rnn'

        # q functions
        self.q_tor = q_tor # uppper max q to define fp
        self.qf = lambda Fx: (.5 * torch.norm(Fx) ** 2) # first term of linearlization
        #self.qf = lambda Fx: (torch.norm(Fx))

        # optimization
        self.iter_fp = iter_fp
        self.iter_noise = iter_noise
        self.lr_fp = lr_fp
        self.lr_decay_tor = lr_decay_tor
        self.lr_decay_epoch = lr_decay_epoch

    def detect_fixedpoints(self, h_init, inputs, ic):
        # Initialize
        fixed_points = {'trajectories':[],'fp':[] ,'epoch':[], 'converge':[]}

        for i in range(ic):
            # Initialize network to one of the trajectories + noise
            # assume 2nd dimension of h_init (list of hidden states) is time
            e, c, qL, fpF, converge, lr_fp = 0, 0, 0 , True, True, self.lr_fp

            # Make Pytorch variable to use autograd
            h_idx = torch.randint(h_init.shape[1], (1,))
            h_local = h_init[:, h_idx, :].detach().clone()
            h_local = h_local + (self.iter_noise ** .5) * torch.randn(h_local.shape)

            # Gradient decent
            while True:
                # Reinstantiate
                h_local = Variable(h_local).to(self.device)
                h_local.requires_grad = True
                h_local.retain_grad()

                # Compute gradient
                q = self.compute_qF(h_local, inputs)
                q.backward()

                # Check convergence status and update learning rate
                # 1) at every lr_decay_epoch, half lr
                # 2) if q value is not changing, half it
                if e % 1000 == 0: print(f'epoch: {e}, q={q.item()}, stuck={c}')
                if e == self.iter_fp: converge = False
                if e % self.lr_decay_epoch == 0 and e > 0: lr_fp *= .5
                if np.abs((q.item() - qL)) < self.lr_decay_tor and e % 500. == 0: lr_fp *= .85; c += 1;

                # Update h_n
                if q.item() < self.q_tor or e == self.iter_fp:break
                h_local = h_local - lr_fp * h_local.grad
                qL = q.item()
                e += 1

            # Evaluate all detected fixed points based on distance
            if i > 0 and converge: fpF = self.eval_fixedpoints(e, fixed_points, h_local)
            if fpF and converge:
                fixed_points['trajectories'].append(torch.squeeze(h_init))
                fixed_points['fp'].append(h_local)
                fixed_points['epoch'].append(e)
                fixed_points['converge'].append(converge)

        return fixed_points

    def compute_qF(self, h_local, inputs):
        _, h_target = getattr(self.model, self.layerN)(inputs, h_local)
        q = self.qf(h_target - h_local) # difference between most updated h_l and h_t
        return q

    def eval_fixedpoints(self, e, fixed_points, h_local, dist_th = .1):
        d = [torch.cdist(f, h_local) for _,f in enumerate(fixed_points['fp'])]
        fpF = torch.all(torch.cat(d) > dist_th) if any(list(map(lambda x:x.nelement(),d))) else False
        return fpF

    def compute_jacobian(self, fixed_points, inputs):
        fixed_points['jacobian'] = []
        fixed_points['eigmode'] = []

        for i, f in enumerate(fixed_points['fp']):
            h = Variable(f).to(self.device)
            h.requires_grad = True
            h.retain_grad()

            # Get the hidden states of the model using at fixed point
            n_units = self.model.n_units_hidden
            _, _h = getattr(self.model, self.layerN)(inputs, h)
            jacobian = torch.zeros(n_units, n_units)

            for i in range(n_units):
                output = torch.zeros(1, 1, n_units).to(self.device)
                output[0, 0, i] = 1. # marks with respect to which unit to compute gradient
                g = torch.autograd.grad(_h, h, grad_outputs = output, create_graph = True)[0]
                jacobian[:, i : i + 1] = einops.repeat(torch.squeeze(g), "i -> i n", n=1)

            # Summarize results
            fixed_points['jacobian'].append(jacobian)

            # Categorizing fixed points
            fixed_points['eigmode'].append(self.eval_stability(jacobian.detach().numpy()))

        return fixed_points

    def eval_stability(self, jacobian):
        # Get jacobian's eigs
        eigv, eigvecs = np.linalg.eig(jacobian)

        # Categorize fixed points
        real = np.abs(eigv)
        fptype = "slow point"
        if np.all(real <= 1.0):fptype = "attractor"
        if np.all(real > 1.0) and np.any(real != 1):fptype = f"{len(np.where(real > 1)[0])}-saddle"

        # Summarize
        eigmode = {"stable": np.all(real <= 1.0), "type":fptype , "eigv":eigv, "eigvec":eigvecs}

        return eigmode
