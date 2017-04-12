import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from multiprocessing.dummy import Pool as ThreadPool
import pdb
import sys

def read_lmdb(lmdb_file, index):

    pool = ThreadPool(4)
    read_para = Read_Render4CNN(lmdb_file)
    results = pool.map(read_para, index)
    pool.close()
    pool.join()
    return results

class Read_Render4CNN(object):

  def __init__(self, lmdb_file):
    self.lmdb_env = lmdb.open(lmdb_file)
  def __call__(self, ind):
    return Read_Once(ind, self.lmdb_env)

def Read_Once(ind, lmdb_env):

    lmdb_txn = lmdb_env.begin(buffers=True)
    datum = caffe_pb2.Datum()
    buf = lmdb_txn.get('%010d'%ind)
    datum.ParseFromString(bytes(buf))
    data = caffe.io.datum_to_array(datum)
    #print 'ind', ind, data.shape
    return data

num = 9000
lmdb_file = '/home/qchen/caffe/data/real_lmdbs/voc12train_all_gt_bbox_lmdb_label'
lmdb_env = lmdb.open(lmdb_file, readonly=True)
print 'num', num
env = lmdb.open('real_label_' + str(num), map_size=1e12)
ind = np.arange(2768, 11768)
#np.random.shuffle(ind)
#ind = ind[np.arange(5000)]
with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(num): 
        print i 
        lmdb_txn = lmdb_env.begin(buffers=True) 
        datum = caffe_pb2.Datum()
        buf = lmdb_txn.get('%010d'%ind[i])
        datum.ParseFromString(buf)
        #flat_x = np.fromstring(datum.data, dtype=np.uint8)
        #y = datum.label      
        #datum1.channels = 
        #pdb.set_trace()
        txn.put('%010d'%i, datum.SerializeToString())

env.close()
env = lmdb.open('real_label_' + str(num))
x = Read_Once(0, env)
print x.shape, 'real_label_' + str(num)

lmdb_file = '/home/qchen/caffe/data/real_lmdbs/voc12train_all_gt_bbox_lmdb_image'
lmdb_env = lmdb.open(lmdb_file, readonly=True)
env = lmdb.open('real_image_' + str(num), map_size=1e12)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(num): 
        print i 
        lmdb_txn = lmdb_env.begin(buffers=True) 
        datum = caffe_pb2.Datum()
        buf = lmdb_txn.get('%010d'%ind[i])
        datum.ParseFromString(buf)
        txn.put('%010d'%i, datum.SerializeToString())

env.close()
env = lmdb.open('real_image_' + str(num))
x = Read_Once(8999, env)
print x.shape, 'real_image_' + str(num)

