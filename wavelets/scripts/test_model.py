import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('finished importing')
#model = sbm.PhaseNet(phases="PSN")
#print(model)

model = torch.load('best_model.pth')
print(model)
model = model.to(device)



data = sbd.STEAD(basepath='/scratch/alpine/mecr8410/PhaseNet/dataset/datasets/stead/', sampling_rate=100)
data = sbd.ETHZ(basepath='/scratch/alpine/mecr8410/PhaseNet/dataset/datasets/ethz/',sampling_rate=100)

#data.preload_waveforms(pbar=True)

#data = sbd.DummyDataset(sampling_rate=100)

train, dev, test = data.train_dev_test()
#test = train

phase_dict = {
        "trace_p_arrival_sample": "P",
        "trace_pP_arrival_sample": "P",
        "trace_P_arrival_sample": "P",
        "trace_P1_arrival_sample": "P",
        "trace_Pg_arrival_sample": "P",
        "trace_Pn_arrival_sample": "P",
        "trace_PmP_arrival_sample": "P",
        "trace_pwP_arrival_sample": "P",
        "trace_pwPm_arrival_sample": "P",
        "trace_s_arrival_sample": "S",
        "trace_S_arrival_sample": "S",
        "trace_S1_arrival_sample": "S",
        "trace_Sg_arrival_sample": "S",
        "trace_SmS_arrival_sample": "S",
        "trace_Sn_arrival_sample": "S",

}

test_generator = sbg.GenericGenerator(test)

augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
        sbg.RandomWindow(windowlen=3001, strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0)

]

test_generator.add_augmentations(augmentations)

#batch_size = 256
batch_size = 1
num_workers = 1  # The number of threads used for loading data

test_loader = DataLoader(test_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_seeding)
print('length test loader:')
print(len(test_loader))
#x = input("stop")


def loss_fn(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    #print(np.sum(y_pred))
    #print(np.sum(y_true))
    #print(y_pred)
    #print(y_true)
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h

def get_accuracy(P_res, S_res):
    P_T1 = P_res.count('TP1') + P_res.count('TN')
    S_T1 = S_res.count('TP1') + S_res.count('TN')
    P_T5 = P_res.count('TP5') + P_res.count('TN') + P_res.count('TP1')
    S_T5 = S_res.count('TP5') + S_res.count('TN') + S_res.count('TP1')
    P_wrong = P_res.count('FP') + P_res.count('FN')
    S_wrong = S_res.count('FP') + S_res.count('FN')
    P_T1_acc = (P_T1) / (P_wrong + P_res.count('TP5') + P_T1)
    S_T1_acc = (S_T1) / (S_wrong + S_res.count('TP5') + S_T1)
    P_T5_acc = (P_T5) / (P_wrong + P_T5)
    S_T5_acc = (S_T5) / (S_wrong + S_T5)

    print('accuracy of p-waves within .1 sec: {}'.format(P_T1_acc), flush=True)
    print('accuracy of s-waves within .1 sec: {}'.format(S_T1_acc), flush=True)
    print('accuracy of p-waves within .5 sec: {}'.format(P_T5_acc), flush=True)
    print('accuracy of s-waves within .5 sec: {}'.format(S_T5_acc), flush=True)

def get_diff_pred(y_true, y_pred, tol, time_diff):
    diff = None
    if torch.max(y_true) > .99 and torch.max(y_pred) > tol:
        diff = time_diff
        if abs(diff) < 10:
            pred = 'TP1'
        elif abs(diff) < 50:
            pred = 'TP5'
        else:
            pred = 'FP'
    elif torch.max(y_true) < .99 and torch.max(y_pred) > tol:
        pred = 'FP'
    elif torch.max(y_true) > .99 and torch.max(y_pred) < tol:
        pred = 'FN'
    elif torch.max(y_true) < .99 and torch.max(y_pred) < tol:
        pred = 'TN'
    return diff, pred

def histogram_data(y_true, y_pred):
    analyst_picks = torch.argmax(y_true, dim=-1)
    alg_picks = torch.argmax(y_pred, dim=-1)
    if torch.max(y_true[0][0]) > .99:
        print("WE HAVE NON-NOISE")
        input('stop')
    #print('----------')
    #print(y_true[0][0])
    #print(y_true[0][0].size())
    #print(analyst_picks)
    #print(alg_picks)
    #print('analyst pick')#
    #print(torch.max(y_true[0][0]))
    #print(torch.max(y_true[0][1]))
    #print('algo pick')#
    #print(torch.max(y_pred[0][0]))
    #print(torch.max(y_pred[0][1]))

    #print('----------')
    time_diffs = analyst_picks-alg_picks
    P_tol = .5
    S_tol = .3
    P_diff, P_pred = get_diff_pred(y_true[0][0], y_pred[0][0], P_tol, time_diffs[0][0].item())
    S_diff, S_pred = get_diff_pred(y_true[0][1], y_pred[0][1], S_tol, time_diffs[0][1].item())
    return [P_pred, S_pred], [P_diff, S_diff]

def test_loop(dataloader):
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()
    P_conf = []
    S_conf = []
    P_time_diff = []
    S_time_diff = []

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))
            #print(pred)
            #print(pred.size())
            #print(batch["y"].to(model.device).size())
            #print(batch['y'].to(model.device))

            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()
            conf_matrix, time_diffs = histogram_data(batch['y'].to(model.device), pred)
            P_conf.append(conf_matrix[0])
            S_conf.append(conf_matrix[1])
            if time_diffs[0]:
                P_time_diff.append(time_diffs[0])
            if time_diffs[1]:
                S_time_diff.append(time_diffs[1])
            #print(P_conf)
            #print(S_conf)
            #print(P_time_diff)
            #print(S_time_diff)
            get_accuracy(P_conf, S_conf)
            #x = input ('stop')
    get_accuracy(P_conf, S_conf)


    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")
    return test_loss

test_loss = test_loop(test_loader)
