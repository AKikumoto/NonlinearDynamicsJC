### PyTourch_data:
# contains functions to generate simualted data sets for PyTourch DataLoader
#
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# ------------------------------------------------------------------------------
### Note:
### LIBRARIES/PACKAGES ---------------------------------------------------------
import numpy as np
import sys
import pandas as pd
import torch
from PIL import Image
import sklearn.datasets # To generate the dataset

### HANDLING DATA --------------------------------------------------------------
class D_BASEDATA(torch.utils.data.Dataset):
    """
    Base class for data class (inherriting Dataset class from PyTorch)
    Define "__len__" and "__getitem__" for using DataLoader
    """
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

    def __len__(self):
       'Method for PyTorch DataLoader: Denotes the total number of samples'
       return None

    def __getitem__(self, index):
        'Method for PyTorch DataLoader:Generates one sample of data'
        # Select sample from list of files
        #ID = self.list_IDs[index]
        #X = self.datasets[ID]
        #y = self.labels[ID]

        # Select sample from randomly simulated data
        x, t = self.simulate(index)
        return x, t

    def simulate(self, index):
        'Method to generate data(x) and target(y)'
        raise NotImplementedError()

    def setloader(self, list_IDs, datasets, datasets_labels):
        'Method for PyTorch DataLoader:List IDs of samples'
        self.list_IDs = list_IDs  #-> ID of datasets
        self.datasets = datasets # -> datasets
        self.datasets_labels = labels #-> label of dayasets
        return None

    def one_hot(self, t):
        t = t.flatten() if t.ndim == 2 else t
        return np.eye(np.max(t) + 1)[t]

    def one_hot_rev(self, t):
        return np.argmax(t, axis = 1)

    def getinfo(self):
        return self.__dict__

### SIMULATED DATA -------------------------------------------------------------
# Spiral data (static)
# n_sample = max number of batch
# n_class = number of classes
# (x) = generated dataset
# (t) = class label of generated data(1 to n_class)
class D_spiral(D_BASEDATA):
    """
    Spiral data (static)
    n_sample = max number of batch
    n_class = number of classes
    (x) = generated dataset
    (t) = class label of generated data(1 to n_class)
    """
    def __init__(self, seed = 612, max_sample = 1000, n_class = 3):
        np.random.seed(seed)
        self.max_sample = max_sample
        self.n_class = n_class

    def __len__(self):
        return self.max_sample

    def simulate(self, index):
        t = np.random.choice(self.n_class, 1)
        rate = index / self.n_sample
        radius = 1.0 * rate
        theta = t * 4.0 + 4.0 * rate + np.random.randn() * 0.2
        x = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
        return x, t # input and target


# Sine wave pattern generation (continuous)
# Using constant frequency value, generate sine waves
# n_sample = max number of batch
# n_time = how many time samples to make
# n_freq = range of frequency of sine wave
# (x) = generated dataset
# (t) = continuous sine wave
class D_sinewave_pattern(D_BASEDATA):
    """
    Sine wave pattern generation (continuous)
    Using constant frequency value, generate sine waves
    n_sample = max number of batch
    n_time = how many time samples to make
    n_freq = range of frequency of sine wave
    (x) = generated dataset
    (t) = continuous sine wave
    """
    def __init__(self, seed = 612, max_sample = 1000, n_time = 50, n_freq = 10):
        # Initialize
        np.random.seed(seed)
        self.max_sample = max_sample
        self.n_time = n_time
        self.n_freq = n_freq

        # Prepare one batch (or called every _getitem__)?
        # self.simulate()

    def __len__(self):
        return self.max_sample

    def simulate(self, index):
        freq = np.random.randint(1, self.n_freq + 1)
        x = np.repeat(freq / self.n_freq + .25, self.n_time)
        x = np.expand_dims(x, axis=1)
        t = np.arange(0, self.n_time * .025, .025)
        t = np.sin(freq * t)
        t = np.expand_dims(t, axis=1)
        return x, t # input and target


# n bit flip flop task (flat)
# Maintain n bit of 3 bit of -1 or +1 state
# n_time = how many time samples to make
# n_rest = fixed time in-between flips
# n_bit = number of bits
# (x) = generated dataset
# (t) = n bits of correct states
class D_nbit_flipflop(D_BASEDATA):
    """
    n bit flip flop task (flat)
    Maintain n bit of 3 bit of -1 or +1 state
    n_time = how many time samples to make
    n_rest = fixed time in-between flips
    n_bit = number of bits
    (x) = generated dataset
    (t) = n bits of correct states
    """
    def __init__(self, seed = 612, max_sample = 1, n_time = 500, n_rest = 50, n_bit = 3):
        # Initialize
        np.random.seed(seed)
        self.n_time = n_time
        self.n_rest = n_rest
        self.n_bit = n_bit
        self.max_sample = max_sample

        # Prepare one batch (or called every _getitem__)?
        # self.simulate()

    def __len__(self):
        return self.max_sample

    def simulate(self):
        self.items = {}
        X_batch = np.zeros((self.n_time, self.n_bit))
        t_batch = np.zeros((self.n_time, self.n_bit))

        for m in range(self.n_bit):
            X, t = np.zeros(self.n_time), np.zeros(self.n_time)

            # Prepare flip flop
            flip_to_p = (np.random.uniform(1, self.n_time - 1, int(self.n_time / self.n_rest))).astype(np.int32)
            flip_to_n = (np.random.uniform(1, self.n_time - 1, int(self.n_time / self.n_rest))).astype(np.int32)
            X[flip_to_p] = 1
            X[flip_to_n] = -1

            # Prepare correct state
            state = 0
            for n, x in enumerate(X):
                if x != 0:state = x
                t[n] = state

            # RNN input: (batch size * n_time * n_input)
            X_batch[:, m] = X.reshape(1, self.n_time, 1).squeeze()

            # RNN target = (batch, seq_len, num_directions * hidden_size)
            t_batch[:, m] = t.reshape(1, self.n_time, 1).squeeze()

        self.items = (X_batch, t_batch)

    def __getitem__(self, index):
        self.simulate() # resample items
        return self.items[0], self.items[1]

### PSYCHOLOGY TASKS -----------------------------------------------------------
# task class must have following methods:
# - initial settings: input and output mapping
# - simulate
# For any psychology tasks with design file
class D_BASETASK(D_BASEDATA):
    def __init__(self, s):
        # Conditions of the task in Task_Design file
        np.random.seed(612)
        for i, (k, v) in enumerate(s.items()):setattr(self, k, v)
        self.design = pd.read_csv(self.taskdir + '/Task_Design.txt', sep="\t")
        self.units = design_to_units(self.design)

        # Dedault settings
        if s.get('n_batch') is None: self.n_batch = 1 # use this to generate data wihtout dataloader
        if s.get('t_label') is None: self.t_label = 'OUTPUT'
        if s.get('units_events') is None: self.units_events = None
        if s.get('noiseF') is None: self.noiseF = lambda x : np.random.normal(0, 1, x)

        # Prepare one batch (or called every _getitem__)?
        #self.data, self.t, self.cl = self.simulate()

    def simulate(self):
        # Initialize
        s, d, u, b = self, self.design, self.units, self.n_batch
        u_on, u_ev, tL = self.units_on, self.units_events, self.timeL

        # Randomly sample trials conditions and correct labels
        cond = np.random.randint(d.shape[0], size = b)
        #cond = np.tile(range(d.shape[0]),int(k['n_batch']/d.shape[0]))
        data = np.vstack((cond.copy(), d[self.t_label][cond]))
        if u_ev is not None: data = data + np.zeros((len(tL), *data.shape))
        if u_ev is not None: data = np.transpose(data,(1, 0, 2))

        # Prepare all requested units (this is somewhat sloww... need to update it!)
        for key in u_on.keys():
            #uID = np.random.randint(u[key].shape[1], size = u_on[key])# Random (note this reqires "pre-sample")
            uID = np.array(range(u[key].shape[1])).repeat(u_on[key])

            for i, v in enumerate(uID):
                a = u[key][cond, v]
                t = np.ones((1, b)) if u_ev is None else np.tile(u_ev[key], (b, 1)).T
                a = a * t # unit pref X activation template
                if u_ev is not None: a = a[np.newaxis, :, :]
                a += self.noiseF(a.shape)
                data = np.concatenate((data, a), axis = 0)

        # Rectangular activatoion
        # 2D output (batch, columns [cond, output, units])
        # 3D output (batch, time samples, columns [cond, output, units])
        data = np.transpose(data)
        self.cdim = data.ndim -1 # last column is always "condition"
        ts = np.take(data, np.array([0, 1]), axis = self.cdim) -1 # -1 for python indexing
        x = np.take(data, range(2, data.shape[self.cdim]), axis = self.cdim)
        return data, ts, x

    def __len__(self):
        'Method for PyTorch DataLoader: Denotes the total number of samples (keep it small)'
        return 2000

    def __getitem__(self, index):
        _, ts , x = self.simulate() # resample items
        cl = np.take(ts, 0, axis = self.cdim) # condition label
        #t = np.take(ts, 1, axis = self.cdim) # target label (repeated over time)
        t = np.unique(np.take(ts, 1, axis = self.cdim)) # one label
        x, t, cl = map(np.squeeze, (x, t, cl))
        t = t[..., np.newaxis] # keep trarget dim fixed
        return x, t, cl


# Takes in the design file of a task and transform to hypothetical units
def design_to_units(design):
    """
    Takes in the design file of a task
    """
    units = {}
    for c in design.columns:
        cond = np.array(design[c])
        elm = np.unique(cond)
        N, C = cond.shape[0], len(elm)

        # Make one hot vector
        one_hot = np.zeros((N, C))
        for i, k in enumerate(elm): one_hot[k == cond,i] = 1
        units[c] = one_hot

    return units

### GRID WORLD  ----------------------------------------------------------------


### FUNCTIONS ------------------------------------------------------------------
# Load (generates) spiral data/task (static)  -> function-based implementation!
# Classify
# n_sample = number of samples in each class
# n_dim = dimension of data (should be 2)
# n_class = number of classes
# (x) = generated dataset
# (t) = class label of generated data(1 to n_class)
def load_spiral(seed=612, n_sample=100, n_dim=2, n_class=3):
    """
    Load (generates) spiral data
    n_sample = number of samples in each class
    n_dim = dimension of output data
    n_class = number of classes
    (x) = generated dataset
    (t) = class label of generated data(1 to n_class)
    """
    np.random.seed(seed)
    x = np.zeros((n_sample*n_class, n_dim))
    t = np.zeros((n_sample*n_class, n_class)) # don't use interger type!

    for j in range(n_class):
        for i in range(n_sample):#N*j, N*(j+1)):
            rate = i / n_sample
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = n_sample*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t

# EXISTING DATASETS ------------------------------------------------------------

# MNIST = image of hand-written digits
# mnist.py will create mnist.pkl

# PTB(Pen Treebank) = text corpus
# ptb.py will creat ptb.pkl

# txtseq = a function converts text sequences
# txtseq.py generates batches, word_to_id, id_to_word

# AirPassanger = 1D time series data
