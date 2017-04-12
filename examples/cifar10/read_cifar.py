import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from multiprocessing.dummy import Pool as ThreadPool
import pdb
import cPickle as pickle
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

lmdb_file = 'cifar10_pool_25000'
lmdb_env = lmdb.open(lmdb_file, readonly=True)
bin = []
for i in range(10):
	bin.append([])

for i in range(25000): 
    print i 
    x, y = Read_Once(i, lmdb_env)
    bin[y].append(i)

# np.save('cifar10_pool_25000/count.npy',count )
# print count

#print bin
pickle.dump(bin,open('bins.p','wb'))
new_bin = pickle.load(open('bins.p','rb'))
print new_bin[0]
print '\n'
print bin[0]