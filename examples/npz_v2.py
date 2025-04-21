from numpy import load
import numpy as np

data = load('/nobackup/rossby27/users/sm_fuxwa/Extreme_Detection/SRGAN_OUT/EPOCH20/preds.npz')
lst = data.files
for item in lst:
    print(item)
    print('type', type(data[item]))
    print('max, min, shape', np.nanmax(data[item]), np.nanmin(data[item]), np.shape(data[item]))
    print(data[item])
