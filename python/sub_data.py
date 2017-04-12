import caffe
import numpy as np
import share_data as sd
#from lmdb_reader import Read_Render4CNN
from lmdb_para import read_lmdb

class Render4CNNLayer_sub(caffe.Layer):

    def setup(self, bottom, top):


        self.data = []
        self.iter = 0
        params = eval(self.param_str_)
        self.source = params['source']
        self.batch_size=params.get('batch_size', 192)
        self.count1 = 0
        self.count2 = 0
        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define 1 top: data or label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        #init batch index
        self.idx = np.arange(self.batch_size)
        self.iidx = []
        self.data = np.array(read_lmdb(self.source, self.idx))
        print self.data.shape

    def reshape(self, bottom, top):
        #print self.idx

        if 'image' in self.source:
            self.data = self.data.reshape(self.batch_size,3,227,227)
            self.data -= sd.imgnet_mean
            top[0].reshape(self.batch_size,3,227,227)
        else:
            top[0].reshape(self.batch_size,4,1,1)
            self.data = self.data.reshape(self.batch_size,4,1,1)


    def forward(self, bottom, top):
        # assign output

        print 'iter',self.iter
        if  (self.iter % 200 < 100):
            self.count1 += 1
            self.idx = np.arange((self.count1-1)*self.batch_size,self.count1 * self.batch_size)%2314401
            if 'image' in self.source:
                if (self.iter % 200 == 0):
                    sd.idx_pool = np.array([])
                sd.idx_pool = np.append(sd.idx_pool, self.idx)
                print 'len of sd.idx_pool'
                print len(sd.idx_pool)
        else:
            if (self.iter % 200 == 100):
                sd.idx = np.argsort(-sd.loss_az - sd.loss_el - sd.loss_t)
                sd.idx_pool = sd.idx_pool[sd.idx]
            self.count2 = self.count2 % 100 + 1
            self.iidx = np.arange((self.count2-1)*self.batch_size,self.count2 * self.batch_size)
            self.idx = np.array(sd.idx_pool[self.iidx])
            #print 'loss',sd.loss_az[sd.idx[self.iidx]]+sd.loss_el[sd.idx[self.iidx]]+sd.loss_t[sd.idx[self.iidx]]
            if (self.iter >= 24000) :
                np.save('ohem_con/sub/mine_data'+str(self.iter)+'.npy',self.idx)
        #print 'idx', self.idx[np.arange(10)]
        self.data = np.array(read_lmdb(self.source, self.idx))
        top[0].data[...] = self.data
        self.iter += 1

    def backward(self, top, propagate_down, bottom):
        pass
