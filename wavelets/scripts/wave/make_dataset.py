import seisbench.data as sbd
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
import pickle


print('finished importing')

data = sbd.STEAD(basepath='/scratch/alpine/mecr8410/PhaseNet/dataset/datasets/stead/', sampling_rate=100)

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

dev_generator.add_augmentations(augmentations)

augmentations = [
        sbg.Scalogram(),
        sbg.ChangeDtype(np.float32),
]

dev_generator.add_augmentations(augmentations)
train_generator.add_augmentations(augmentations)
batch_size = 1
num_workers = 1  # The number of threads used for loading data

train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)
dev_loader = DataLoader(dev_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)
test_loader = DataLoader(test_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)

data_dict = {'train': {'X': [], 'y': []},
             'val':   {'X': [], 'y': []},
             'test':  {'X': [], 'y': []}}

for batch in train_loader:
    data_dict['train']['X'].append(batch["X"])
    data_dict['train']['y'].append(batch["y"])

for batch in dev_loader:
    data_dict['val']['X'].append(batch["X"])
    data_dict['val']['y'].append(batch["y"])

for batch in test_loader:
    data_dict['test']['X'].append(batch["X"])
    data_dict['test']['y'].append(batch["y"])

with open('stead_scaleo_dataset.pickle', 'wb') as handle:
        pickle.dump(wf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
