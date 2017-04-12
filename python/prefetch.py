import sys
import numpy as np
from lmdb_para import *
import os
train_image_folder = 'data/syn_lmdbs/syn_lmdb_train_image_rand'
train_label_folder = 'data/syn_lmdbs/syn_lmdb_train_label_rand'
import pdb
import matplotlib.pyplot as plt
ind = np.load(sys.argv[1])
np_folder = sys.argv[2]
ind_new = np.where(ind < 0)[0]
ind_old = np.where(ind >= 0)[0]
if len(ind_new) :
	ind_new = ind[ind_new]
	ind_new = -ind_new - 1
	data_new = np.load(os.path.join(np_folder, 'image.npy'))
	data_new = data_new[ind_new]
	label_new = np.load(os.path.join(np_folder, 'label.npy'))
	label_new = label_new[ind_new]


if len(ind_old):
	ind_old = ind[ind_old]
	data_old = read_lmdb(train_image_folder, ind_old)
	data_old = np.asarray(data_old)
	data_old = data_old.reshape((len(ind_old), 3, 227, 227))
	data_old = data_old.astype(np.uint8)
	
	label_old = read_lmdb(train_label_folder, ind_old)
	label_old = np.asarray(label_old)
	label_old = label_old.reshape((len(ind_old), 4, 1, 1))
	label_old = label_old.astype(np.uint16)

if (len(ind_new) and len(ind_old)):
	data = np.row_stack((data_new, data_old))
	label = np.row_stack((label_new, label_old))
elif len(ind_new):
	data = np.asarray(data_new)
	label = np.asarray(label_new)
elif len(ind_old):
	data = np.asarray(data_old)
	label = np.asarray(label_old)
#pdb.set_trace()
data = data.reshape(len(ind), 3, 227,227)
label = label.reshape(len(ind), 4, 1,1)
np.save(os.path.join(np_folder,sys.argv[3]+'_image.npy'), data)
np.save(os.path.join(np_folder,sys.argv[3]+'_label.npy'), label)

# if len(ind_new) :
# 	ind_new = ind[ind_new]
# 	ind_new = -ind_new - 1
# 	data_new = np.load(os.path.join(np_folder, 'image.npy'))
# 	for i in ind_new:
# 		im = data_new[ind_new]
# 		im = np.transpose(im, (1, 2, 0))
# 		plt.imshow(im)
# 		print 'new'
# 		plt.show()


# if len(ind_old):
# 	ind_old = ind[ind_old]
# 	for i in ind_old:
# 		data_old = read_lmdb(train_image_folder, [i])
# 		im = (data_old.astype(np.uint8)).reshape(3,227,227)
# 		im = np.transpose(im, (1, 2, 0))
# 		plt.imshow(im)
# 		print 'old'
# 		plt.show()
# 		
