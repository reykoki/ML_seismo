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

batch_size = 256
#batch_size = 4
num_workers = 4  # The number of threads used for loading data
#batch_size = 1
#num_workers = 1  # The number of threads used for loading data

train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)
dev_loader = DataLoader(dev_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)

learning_rate = 1e-2
epochs = 100

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def loss_fn(y_pred, y_true, eps=1e-5):
    # exclude noise
    y_pred = y_pred[0][0:2]
    y_true = y_true[0][0:2]
#    # vector cross entropy loss
#    print('----------------')
#    print(y_true[0])
#    print(y_pred[1])
#    print(torch.max(y_true[0]).item())
#    print(torch.max(y_true[1]).item())
#    print(torch.max(y_pred[0]).item())
#    print(torch.max(y_pred[1]).item())
#
#    print(torch.sum(y_pred[0]).item())
#    print(torch.sum(y_pred[1]).item())
#    print(torch.sum(y_pred).item())
#    print(torch.sum(y_true[0]).item())
#    print(torch.sum(y_true[1]).item())
#
#    analyst_picks = torch.argmax(y_true, dim=-1)
#    alg_picks = torch.argmax(y_pred, dim=-1)
#    print(analyst_picks)
#    print(alg_picks)
#    print('----------------')
#    if torch.sum(y_true[0]).item() < .01 and torch.sum(y_true[1]).item() < .01:
#        print('NO PICK!!')
#    else:
#        x = input('stop')
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


def train_loop(dataloader):
    size = len(dataloader.dataset)
    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(batch["X"].to(model.device))
        loss = loss_fn(pred, batch["y"].to(model.device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 5 == 0:
            loss, current = loss.item(), batch_id * batch["X"].shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))
            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")
    return test_loss

best_loss = 1e10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader)
    test_loss = test_loop(dev_loader)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model, 'best_model.pth')

