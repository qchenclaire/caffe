import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from trial_2_share import *
#lmdb_file = '../data/Su/syn_lmdb_train_image'
#lmdb_file1 = './data/syn_lmdbs/syn_lmdb_test_label'
def Read_Render4CNN(lmdb_file, index):

    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin(buffers=True)
    datum = caffe_pb2.Datum()
    data = []
    for ind in index:
        #print 'ind',ind
        buf = lmdb_txn.get('%010d'%ind)
        datum.ParseFromString(bytes(buf))
        tmp = caffe.io.datum_to_array(datum)
        data = np.append(data,tmp)
        #im = data.astype(np.uint8)
        #im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
        #print "index ", ind
        #print "data", data

    return data
    #lmdb_cursor = lmdb_txn.cursor()

    #for key, value in lmdb_cursor:
    #     datum.ParseFromString(value)
    #
    #     index = datum.label
    #     data = caffe.io.datum_to_array(datum)
    #     #im = data.astype(np.uint8)
    #     #im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
    #     print "index ", index
    #     print "data", data
    #     ##plt.imshow(im)
    #     #plt.show()
#data = Read_Render4CNN(lmdb_file,[1])

#im = data.astype(np.uint8).reshape(3,227,227)
#a = (im - imgnet_mean)[:]

#print im.shape
#im = np.transpose(im, (1, 2, 0))
#plt.imshow(im)
#plt.show()
