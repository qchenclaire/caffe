import numpy as np
import sys
import lmdb
import caffe
import cPickle as pickle
import pdb
from caffe.proto import caffe_pb2
iter = 63000
def Read_Once(ind, lmdb_env):

    lmdb_txn = lmdb_env.begin(buffers=True)
    datum = caffe_pb2.Datum()
    buf = lmdb_txn.get('%05d'%ind)
    datum.ParseFromString(buf)
    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label
    return x, y
class Data_hard(caffe.Layer):

    def setup(self, bottom, top):


        self.data = []
        self.label = []
        self.iter = iter
        print 'iter', self.iter
        sys.stdout.flush()
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define 2 top: data or label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")


    def reshape(self, bottom, top):
        #print self.idx
        top[0].reshape(20,3,32,32)
        top[1].reshape(20)


    def forward(self, bottom, top):
        # assign output

        if  (self.iter  < 56000)and (self.iter%2000 < 500) :
            data = np.load('examples/cifar10/mine/data/cur_data.npy')
            label = np.load('examples/cifar10/mine/data/cur_label.npy')
        else:
            data = np.load('examples/cifar10/mine/data/cummulated_data.npy')
            label = np.load('examples/cifar10/mine/data/cummulated_label.npy')
        self.idx = np.arange(len(data))
        np.random.shuffle(self.idx)
        self.idx = self.idx[np.arange(20)]
        self.data = data[self.idx]
        self.label = label[self.idx]

        top[0].data[...] = self.data
        top[1].data[...] = self.label
        self.iter += 1

    def backward(self, top, propagate_down, bottom):
        pass

class Accuracy(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")
        self.count = np.zeros(10, dtype = np.float)
        self.acc = np.zeros(10, dtype = np.float)
        self.iter = iter
        self.f = open('examples/cifar10/mine/test/acc.txt','w')
        self.g = open('examples/cifar10/mine/guide/acc.txt','w')
        self.mean = np.load('examples/cifar10/mean.npy')
        self.mean = self.mean.astype(np.int)

    def reshape(self, bottom, top):

        if (bottom[0].data.size / bottom[0].num != 10 ):
            raise Exception("number of classes != 10.")
        top[0].reshape(1)

    def forward(self, bottom, top):

        # assign outputi

        accuracy = float(0)
        count = float(0)

        p = np.array(bottom[0].data)
        gt = np.array(bottom[1].data).reshape(bottom[0].num).astype(int)
        self.count = np.zeros(10)
        self.acc = np.zeros(10)
        for i in range(10000):

            p_i = p[i]
            pred_i = np.argmax(p_i)
            gt_i = gt[i]
            self.count[gt_i] +=1
            if gt_i == pred_i:
                self.acc[gt_i] += 1
                accuracy += 1
            count += 1

        top[0].data[0] = accuracy / count
        tmp = self.acc/self.count
        np.save('examples/cifar10/mine/test/acc' + str(self.iter) + '.npy', tmp)
        self.f.write('%f\n' %top[0].data[0])


        if (self.iter < 56000) and (self.iter%2000==0):
            self.count = np.zeros(10)
            self.acc = np.zeros(10)
            accuracy = float(0)
            count = float(0)
            for i in range(10000, 15000):

                p_i = p[i]
                pred_i = np.argmax(p_i)
                gt_i = gt[i]
                self.count[gt_i] += 1
                if gt_i == pred_i:
                    self.acc[gt_i] += 1
                    accuracy += 1
                count += 1

            tmp = self.acc / self.count

            np.save('examples/cifar10/mine/guide/acc' + str(self.iter) + '.npy', tmp)
            self.g.write('%f\n' % (accuracy / count))

            dist = np.exp(-3 * tmp)
            dist = dist / sum(dist)
            np.save('examples/cifar10/mine/guide/dist' + str(self.iter) + '.npy', dist)
            samples = np.random.choice(10, 1000, p=dist)
            unique, counts = np.unique(samples, return_counts=True)
            pairs = dict(zip(unique, counts))
            bins = pickle.load(open('examples/cifar10/cifar10_pool_25000/bins.p', 'rb'))
            lmdb_file = 'examples/cifar10/cifar10_pool_25000'
            lmdb_env = lmdb.open(lmdb_file, readonly=True)
            xs = []
            ys = []
            for j in pairs.keys():
                idx = np.arange(len(bins[j]))
                if pairs[j] <= len(bins[j]):
                    idx = idx[np.arange(pairs[j])]
                for k in idx:
                    ind = bins[j][k]
                    x, y = Read_Once(ind, lmdb_env)
                    x = x.astype(np.int)
                    x -= self.mean
                    xs.append(x)
                    ys.append(y)
                for k in idx:
                    del bins[j][0]
            while len(ys) < 1000:
                count = np.zeros(10)
                for n in range(10):
                    count[n] = len(bins[n])
                n = np.argmax(count)
                ind = bins[n][0]
                x, y = Read_Once(ind, lmdb_env)
                x = x.astype(np.int)
                x -= self.mean
                xs.append(x)
                ys.append(y)
                del bins[n][0]
            pickle.dump(bins, open('examples/cifar10/cifar10_pool_25000/bins.p', 'wb'))
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            np.save('examples/cifar10/mine/data/cur_data.npy', xs)
            np.save('examples/cifar10/mine/data/cur_label.npy', ys)
            if self.iter == 36000:
                np.save('examples/cifar10/mine/data/cummulated_data.npy', xs)
                np.save('examples/cifar10/mine/data/cummulated_label.npy', ys)
            else:
                cum_data = np.load('examples/cifar10/mine/data/cummulated_data.npy')
                cum_label = np.load('examples/cifar10/mine/data/cummulated_label.npy')
                cum_data = np.row_stack((cum_data,xs))
                cum_label = np.append(cum_label, ys)
                print 'cum_data', cum_data.shape
                sys.stdout.flush()

                np.save('examples/cifar10/mine/data/cummulated_data.npy', cum_data)
                np.save('examples/cifar10/mine/data/cummulated_label.npy', cum_label)



        self.iter += 1000

    def backward(self, top, propagate_down, bottom):
        pass