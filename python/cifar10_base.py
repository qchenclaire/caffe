import numpy as np
import sys

import caffe

class Identity(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].data.size != bottom[1].data.size:
            raise Exception("CONCAT DIFFERENT TYPES OF DATA.")


    def reshape(self, bottom, top):

        pass
        top[0].reshape(100,3,32,32)

    def forward(self, bottom, top):

        top[0].data[...] = np.append(bottom[0].data,bottom[1].data)

    def backward(self, top, propagate_down, bottom):
        pass

class Accuracy(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")
        self.count = np.zeros(10, dtype = np.float)
        self.acc = np.zeros(10, dtype = np.float)
        self.iter = 24000

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
        for i in range(bottom[0].num):

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
        for i in range(1):
            sys.stdout.write('class %d, acc %d' %(i, tmp[i]))
            sys.stdout.flush()
        np.save('examples/cifar10/random/acc' + str(self.iter) + '.npy', tmp)
        self.iter += 1000

    def backward(self, top, propagate_down, bottom):
        pass
