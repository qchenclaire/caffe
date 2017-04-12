import caffe
import numpy as np
import random
import os, struct, sys
#sys.path.append('/mnt/disk1/RenderForCNN')
#sys.path.append('/mnt/disk1/RenderForCNN/render_pipeline')
#from haosu_realtime import *
import share_data as sd
from compute_acc import *

class Data(caffe.Layer):

    def setup(self, bottom, top):


        self.data = []
        self.label = []
        self.iter = 3804
        #self.count = np.load("python/accum.npy")
        #params = eval(self.param_str)
        #self.source = params['source']
        #self.init = params.get('init', True)
        #self.seed = params.get('seed', None)
        #self.batch_size = params.get('batch_size', 192)
        self.batch_size = 192
        self.model_idx = []
        self.view_params = []

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define 2 tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")


    def reshape(self, bottom, top):

        top[0].reshape(self.batch_size,3,227,227)
        top[1].reshape(self.batch_size,4,1,1)


    def forward(self, bottom, top):
        # assign output
        self.model_idx, self.view_params = sample_view(views_all)
        print 'model_idx', self.model_idx.shape, 'view_params',self.view_params.shape
        render_one_batch(self.model_idx, self.view_params, self.iter)
        self.data, label = crop_overlay_batch(self.iter)
        self.label = label[:,1:5]
        self.label = self.label.reshape(self.batch_size,4,1,1)

        self.data = self.data.reshape(self.batch_size,3,227,227)
        self.data -= sd.imgnet_mean

        top[0].data[...] = self.data
        top[1].data[...] = self.label

        self.iter += 1


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
#	print 'num', num
#	print 'bottom1.num',bottom[1].data.shape
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
                 #for k in range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period):
		 #    print 'self.data', self.prob_data[k]
                 probs_cls_sum = np.float32(np.sum(self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)]))
		 #print 'prob_cls_sum',probs_cls_sum
                 self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)] /= probs_cls_sum
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
        #f = open('experiment/vgg16_baseline/loss.txt','a')
	#f.write(str(loss)+'\n')
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

        accuracy = float(0)

        p_az = np.array(bottom[0].data)
        p_el = np.array(bottom[1].data)
        p_t = np.array(bottom[2].data)

        az = np.array(bottom[3].data).reshape(bottom[0].num).astype(int)
        el = np.array(bottom[4].data).reshape(bottom[0].num).astype(int)
        t = np.array(bottom[5].data).reshape(bottom[0].num).astype(int)

        loss = np.array(bottom[6].data + bottom[7].data + bottom[8].data)/3.0
        #sd.hot_idx = np.argsort(-loss)[0:24]
        #sd.model_idx = sd.model_idx[sd.hot_idx]
        #sd.view_params = sd.view_params[sd.hot_idx]
        #model_idx = np.load('realtime_cache/model_idx.npy')
        #model_idx = np.append(model_idx, sd.model_idx)
        #np.save('realtime_cache/model_idx.npy', model_idx)
        #view_params = np.load('realtime_cache/view_params.npy')
        #view_params = np.append(model_idx, sd.model_idx)
        #np.save('realtime_cache/view_params.npy', view_params)

        az[az >= 10000] -= 10000
        el[el >= 10000] -= 10000
        t[t >= 10000] -= 10000

        az360 = np.array(az % (self.period))
        el360 = np.array(el % (self.period))
        t360 = np.array(t % (self.period))

        cls_idx = az / (self.period)
        nonbkg_cnt = np.count_nonzero(cls_idx!=12)

         #= np.zeros(bottom[0].num,self.period)
        for i in range(bottom[0].num):
            if cls_idx[i] == 12:
                continue
            tmp_az = p_az[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            in_az = tmp_az[np.arange(az360[i]-15, az360[i] + 16)%360]
            out_az = tmp_az[np.arange(az360[i]+16, 360+az360[i] -15)%360]
            pred_az = np.argmax(tmp_az)

            tmp_el = p_el[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            in_el = tmp_el[np.arange(el360[i]-5, el360[i] + 6)%360]
            out_el = tmp_el[np.arange(el360[i]+6, 360+el360[i] -5)%360]
            pred_el = np.argmax(tmp_el)

            tmp_t = p_t[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            in_t = tmp_t[np.arange(t360[i]-5, t360[i] + 6)%360]
            out_t = tmp_t[np.arange(t360[i]+6, 360+t360[i] -5)%360]
            pred_t = np.argmax(tmp_t)

            det = compute_acc([pred_az, pred_el, pred_t],[az360[i], el360[i], t360[i]])
            #print 'det',det
            accuracy += (det < math.pi / 6.0)

        top[0].data[0] = accuracy / nonbkg_cnt
        self.iter += 1

    def backward(self, top, propagate_down, bottom):
        pass
