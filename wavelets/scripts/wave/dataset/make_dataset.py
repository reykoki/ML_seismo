import seisbench.data as sbd
import h5py
import pickle
import torch.nn.functional as F
from tabulate import tabulate
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import time


print('finished importing')

device = torch.device("cpu")

data = sbd.STEAD(phases='PSN', cache='full')
data.preload_waveforms(pbar=True)

train, dev, test = data.train_dev_test()

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_s_arrival_sample": "S"
}

train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)
test_generator = sbg.GenericGenerator(test)

augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
        sbg.RandomWindow(windowlen=3001, strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0)
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)
test_generator.add_augmentations(augmentations)

batch_size = 1
num_workers = 0  # The number of threads used for loading data

train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)
test_loader = DataLoader(test_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)
dev_loader = DataLoader(dev_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)

f = h5py.File("ds_wf.hdf5", "w")

def data_loop(dataloader, cat, f, num_samples=None):
    if not num_samples:
        num_samples = len(dataloader)
    X = np.empty((num_samples, 3, 3001))
    y = np.empty((num_samples, 3, 3001))
    for batch_id, batch in enumerate(dataloader):
        data = batch["X"].to(device)[0].detach().cpu().numpy()
        truth = batch["y"].to(device)[0].detach().cpu().numpy()
        X[batch_id] = data
        y[batch_id] = truth
        if batch_id == num_samples-1:
            break
    d_type = f.create_group(cat)
    ds_X = d_type.create_dataset('X', shape=(num_samples,3,3001), data = X, dtype=np.float32)
    ds_y = d_type.create_dataset('y', shape=(num_samples,3,3001), data = y, dtype=np.float32)
    return f

to = time.time()
f = data_loop(train_loader, 'train', f)
tf = time.time()
print('time for train: ', tf-to)

to = time.time()
f = data_loop(dev_loader, 'val', f)
tf = time.time()
print('time for val: ', tf-to)

to = time.time()
f = data_loop(test_loader, 'test', f)
tf = time.time()
print('time for test: ', tf-to)

f.close()


