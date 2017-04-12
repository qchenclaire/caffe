# 3/2/2017 80% Su + 20% Hard
import struct
from trial_4_helper import *
from compute_acc import *
import trial_2_share as sd
from lmdb_para import *
import pdb
def find_interval(azimuth, a):
    for i in range(len(a)):
        if azimuth < a[i]:
            break
    ind = i -1
    if azimuth > a[-1]:
        ind = 0
    return ind
class Data(caffe.Layer):

    def setup(self, bottom, top):
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define 2 tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        self.data = []
        self.data_real = []
        self.data_syn = []
        self.label = []
        self.label_real = []
        self.label_syn = []
        self.iter = 0
        params = eval(self.param_str)
        self.syn_image_folder = params['syn_image_folder']
        self.syn_label_folder = params['syn_label_folder']
        self.real_image_folder = params['real_image_folder']
        self.real_label_folder = params['real_label_folder']
        self.hard_folder = params['hard_folder']
        self.batch_size = params.get('batch_size', 256)
        self.batch_size_syn = self.batch_size

        self.syn_idx = np.arange(self.batch_size_syn)
        self.syn_ind = np.arange(2142100)
        self.syn_sub_ind = np.arange(self.batch_size_syn)


    def reshape(self, bottom, top):

        top[0].reshape(self.batch_size,3,227,227)
        top[1].reshape(self.batch_size,4,1,1)


    def forward(self, bottom, top):

        self.syn_idx = self.syn_ind[self.syn_sub_ind]
        self.data_syn = np.array(read_lmdb(self.syn_image_folder, self.syn_idx))
        self.label_syn = np.array(read_lmdb(self.syn_label_folder, self.syn_idx))

        if 2142099 in self.syn_sub_ind:
            np.random.shuffle(self.syn_ind)
        self.syn_sub_ind = (self.syn_sub_ind + self.batch_size_syn) % 2142100

        self.data = self.data_syn
        self.label = self.label_syn

        self.label = self.label.reshape(self.batch_size,4,1,1)
        self.data = self.data.reshape(self.batch_size,3,227,227)
        self.data = self.data.astype(np.int)
        self.data -= sd.imgnet_mean

        top[0].data[...] = self.data
        top[1].data[...] = self.label

    def backward(self, top, propagate_down, bottom):
        pass

class SoftmaxViewLoss(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute loss.")

        self.weights_sum = 0
        self.prob_data = []
        self.label=[]
        self.weight = []
        self.dim = bottom[0].data.size / bottom[0].num
        self.spatial_dim = 0
        self.cls_idx = []
        self.diff = []
        self.iter = 0
        self.ind = []
        self.cnt = 0
        self.loss =[]
        self.count = [0]
        self.mine = []

        params = eval(self.param_str)
        self.type_ = params['type']
        self.bandwidth = params.get('bandwidth', 5)
        self.sigma = np.float32(params.get('sigma', 3))
        self.pos_weight=params.get('pos_weight', 1)
        self.neg_weight=params.get('neg_weight', 0)
        self.period=int(params.get('period', 360))
        for k in range(-self.bandwidth, self.bandwidth+1):
            self.weights_sum += np.exp(-abs(k)/self.sigma)



    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")

        top[1].reshape(bottom[0].num)
        top[2].reshape(*bottom[0].data.shape)
        top[0].reshape(1)

    def forward(self, bottom, top):

        num = bottom[0].num
        self.loss = np.zeros(num, dtype = np.float)
        dim = 4320
        scores = np.array(bottom[0].data)
        tmp = np.tile(np.max(scores,axis=1),4320)
        tmp = tmp.reshape(scores.T.shape).T
        scores -= tmp
        exp_scores = np.exp(scores)
        self.prob_data = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.prob_data = self.prob_data.reshape(num*4320)


        self.spatial_dim = bottom[0].height * bottom[0].width
        if (self.spatial_dim != 1):
            raise Exception("self.spatial_dim != 1")

        self.label = np.array(bottom[1].data).reshape(num)
        self.label = self.label.astype(int)

        weight = np.float32(0)
        loss = np.float32(0)
        nonbg = 0
        for i in range(num):
             for j in range(self.spatial_dim):
                 label_value = self.label[i * self.spatial_dim + j]
                 if (label_value < 10000):
                     weight = self.pos_weight
                 else :
                     weight = self.neg_weight
                     label_value = label_value - 10000

          # Added by rqi, full of HARD CODING..
          # ASSUMPTION: classes number is 12*360 or 12*360 + 1
                 cls_idx = int(label_value / self.period )# 0~11,12->bkg
                 if (cls_idx == 12) : continue # no loss for bkg
                 nonbg += 1
                 probs_cls_sum = np.float32(np.sum(self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)]))
                 #print 'before'
		 #print self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)]
                 self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)] /= probs_cls_sum
                 #print 'after'
		 #print self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)]
                # convert to 360-class label
                 view_label = label_value % self.period
                 tmp_loss = np.float32(0)
                 for k in range(-self.bandwidth, self.bandwidth+1):
                     view_k = (view_label + k) % self.period
                    # convert back to 4320-class label
                     label_value_k = view_k + cls_idx * self.period

                     tmp_loss -= np.exp(-abs(k)/self.sigma) * np.log(max(self.prob_data[i * dim + label_value_k * self.spatial_dim + j],10**(-37)))

                # scale loss with loss weight
                 self.loss[i] += tmp_loss * weight

        loss = np.sum(self.loss) / nonbg
        top[0].data[...] = loss

        top[1].data[...] = self.loss.reshape(num)
        top[2].data[...] = self.prob_data.reshape(num,dim)


    def backward(self, top, propagate_down, bottom):
        num = bottom[0].num
        dim = 4320
        if propagate_down[1]:
            raise Exception("Layer cannot backprop to self.label inputs.")
        if propagate_down[0]:
            diff = np.zeros_like(self.prob_data)
            for i in range(num):
                for j in range(self.spatial_dim):
                    label_value = self.label[i * self.spatial_dim + j]
                    if (label_value < 10000):
                        weight = self.pos_weight
                    else :
                        weight = self.neg_weight
                        label_value = label_value - 10000
                # Added by rqi, full of HARD CODING..
                # ASSUMPTION: classes number is 12*360 or 12*360 + 1
                    cls_idx = label_value / self.period # 0~11,12->bkg
                    diff[range(i*dim + cls_idx*self.period, i*dim + cls_idx*self.period + self.period)] = self.prob_data[range(i*dim + cls_idx*self.period, i*dim + cls_idx*self.period + self.period)] * self.weights_sum
                    diff[range(i*dim,i*dim+dim)] *= weight
                    view_label = label_value % self.period
                    for k in range(-self.bandwidth, self.bandwidth+1):
                        # e.g. view_label+k=-3 --> 357
                        view_k = (view_label + k) % self.period
                        # convert back to 4320-class label
                        label_value_k = view_k + cls_idx * self.period
                        diff[i * dim + label_value_k * self.spatial_dim + j] -= np.exp(-abs(k)/self.sigma) * weight

            diff = diff.reshape(bottom[0].data.shape)
            bottom[0].diff[...] = diff / (num / 8) / self.spatial_dim

class Accuracy(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")
        self.iter = 0
        self.period = int(360)
        #np.save('realtime_cache/model_idx.npy', np.array([],dtype = np.int))
        #np.save('realtime_cache/view_params.npy', np.array([],dtype = np.int))


    def reshape(self, bottom, top):
        #print bottom[0].data.size,
        if (bottom[0].data.size / bottom[0].num != self.period * 12 ):
            raise Exception("number of classes != 4320.")
        top[0].reshape(1)

    def forward(self, bottom, top):

        top[0].data[0] = 0
        self.iter += 1

    def backward(self, top, propagate_down, bottom):
        pass
