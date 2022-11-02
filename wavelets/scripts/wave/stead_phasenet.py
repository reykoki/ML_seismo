import seisbench.data as sbd
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


print('finished importing')
model = sbm.PhaseNet(phases="PSN", in_channels=96)
print(model)

model.cuda();

data = sbd.STEAD(cache='full')
data.preload_waveforms(pbar=True)


train, dev, test = data.train_dev_test()

#meta = data.metadata
#meta.to_csv('meta_stead.txt', sep='\t', index=False)
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

scalo = [
        sbg.Scalogram(),
        sbg.ChangeDtype(np.float32),
]

train_generator.add_augmentations(scalo)
dev_generator.add_augmentations(scalo)

batch_size = 512
num_workers = 4  # The number of threads used for loading data
#batch_size = 1
#num_workers = 1  # The number of threads used for loading data

train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_seeding)
dev_loader = DataLoader(dev_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)

learning_rate = 1e-2
epochs = 100

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def loss_fn(y_pred, y_true):
    y_pred = y_pred.float()
    y_true = y_true.float()
    loss = F.binary_cross_entropy_with_logits(y_true, y_pred)
    return loss

def train_loop(dataloader):
    size = len(dataloader.dataset)
    total_loss = 0.0
    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(batch["X"].to(model.device))
        loss = loss_fn(pred, batch["y"].to(model.device))
        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    print("Training Loss:   {0}".format(round(epoch_loss,8)), flush=True)

def val_loop(dataloader):
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))
            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()
    test_loss /= num_batches
    print(f"Validation loss: {test_loss:>8f} \n")
    return test_loss

best_loss = 1e10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader)
    test_loss = val_loop(dev_loader)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model, 'models/stead_best_model.pth')

