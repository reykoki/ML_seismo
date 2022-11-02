import h5py
import pywt
import scipy
import numpy as np
import time

print('finished importing')
wf_fn = 'stead_wf_ds.hdf5'

wf_ds = h5py.File(wf_fn, 'r')

def get_scaleo(wfs, scale_lim=33, waveletname = 'mexh'):
    scales = np.arange(1, scale_lim)
    all_powers = []
    for wf in wfs:
        [coefficients, freq] = pywt.cwt(data=wf,
        scales=scales,
        wavelet=waveletname,
        sampling_period=1/100)
        power = np.log2(abs(coefficients))
        norm_power= (power - np.min(power)) / (np.max(power) - np.min(power))
        all_powers.extend(np.array(norm_power))
    return np.array(all_powers)


scaleo_ds = h5py.File("ds_scaleo.hdf5", "w")

def get_data(scaleo_ds, cat):
    for idx, X in enumerate(wf_ds[cat]['X']):
        scaleo_ds[cat]['y'].append(wf_ds[cat]['y'][idx])
        scaleo_ds[cat]['X'].append(get_scaleo(X))
    return scaleo_ds


to = time.time()
scaleo_ds = get_data(scaleo_ds, 'train')
tf = time.time()
print('time for train: ', tf-to)

to = time.time()
scaleo_ds = get_data(scaleo_ds, 'val')
tf = time.time()
print('time for val: ', tf-to)

to = time.time()
scaleo_ds = get_data(scaleo_ds, 'test')
tf = time.time()
print('time for test', tf-to)





