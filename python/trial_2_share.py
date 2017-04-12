import numpy as np
mode = 'val'
render_size = 0
val_epoch = 0
train_iter = 0
imgnet_mean=np.ndarray(shape=(3,227,227),dtype=np.uint8)
imgnet_mean[0] = 104
imgnet_mean[1] = 117
imgnet_mean[2] = 123