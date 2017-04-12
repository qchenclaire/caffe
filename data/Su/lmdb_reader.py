import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
import sys
lmdb_file = 'syn_lmdb_test_image'
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
num = int(sys.argv[1])
data = Read_Render4CNN(lmdb_file,[num])

im = data.astype(np.uint8).reshape(3,227,227)


print im.shape
im = np.transpose(im, (1, 2, 0))
plt.imshow(im)
plt.show()

#a = [0,2,3, 5,6,7,8, 12,15, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 35, 36, 37, 38, 39, 40]
a = np.load('tmp.npy')
#a = a[:-1]
a = np.append(a,num)
np.save('tmp',a)
print len(a)