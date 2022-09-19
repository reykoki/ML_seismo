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
model = sbm.PhaseNet(phases="PSN")
print(model)

model.cuda();

data = sbd.STEAD(basepath='/scratch/alpine/mecr8410/PhaseNet/dataset/datasets/stead/', sampling_rate=100)
data.preload_waveforms(pbar=True)

train, dev, test = data.train_dev_test()

meta = data.metadata
meta.to_csv('meta_stead.txt', sep='\t', index=False)
phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_s_arrival_sample": "S"
}

train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(dev)

augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
        sbg.RandomWindow(windowlen=3001, strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0)

]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)


wf_dict = {}

for i in range(250):
    rand_int = np.random.randint(len(train_generator))
    sample = train_generator[rand_int]
    wf_dict.update({rand_int: {'data': sample['X'], 'truth': sample['y'] }})

with open('stead_samples.pickle', 'wb') as handle:
        pickle.dump(wf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
