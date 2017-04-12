import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from multiprocessing.dummy import Pool as ThreadPool
import pdb

def read_lmdb(lmdb_file, index):

    pool = ThreadPool(4)
    read_para = Read_Cifar(lmdb_file)
    xs, ys = pool.map(read_para, index)
    pool.close()
    pool.join()
    return xs, ys

class Read_Cifar(object):

  def __init__(self, lmdb_file):
    self.lmdb_env = lmdb.open(lmdb_file, readonly=True)
  def __call__(self, ind):
    return Read_Once(ind, self.lmdb_env)

def Read_Once(ind, lmdb_env):

    lmdb_txn = lmdb_env.begin(buffers=True)
    datum = caffe_pb2.Datum()
    buf = lmdb_txn.get('%05d'%ind)
    datum.ParseFromString(buf)
    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label
    return x, y

lmdb_file = 'cifar10_train_lmdb'
lmdb_env = lmdb.open(lmdb_file, readonly=True)

env = lmdb.open('cifar10_random_5000', map_size=1e12)
ind = np.arange(5000,10000)
#np.random.shuffle(ind)
#ind = ind[np.arange(5000)]
with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(5000): 
        print i 
        lmdb_txn = lmdb_env.begin(buffers=True) 
        datum = caffe_pb2.Datum()
        buf = lmdb_txn.get('%05d'%ind[i])
        datum.ParseFromString(buf)
        #flat_x = np.fromstring(datum.data, dtype=np.uint8)
        #y = datum.label      
        #datum1.channels = 
        #pdb.set_trace()
        txn.put('%05d'%i, datum.SerializeToString())

env.close()
env = lmdb.open('cifar10_random_5000')
x, y = Read_Once(0, env)
print x.shape
print y


