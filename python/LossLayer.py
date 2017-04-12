import numpy as np
import caffe
import ipdb

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

        params = eval(self.param_str_)
        print params
        self.type_ = params['type']
        self.bandwidth = params.get('bandwidth', 5)
        self.sigma = float(params.get('sigma', 3))
        self.pos_weight=params.get('pos_weight', 1)
        self.neg_weight=params.get('neg_weight', 0)
        self.period=int(params.get('period', 360))



        self.weights_sum = np.sum(np.exp(-abs(np.arange(-self.bandwidth, self.bandwidth+1))/ self.sigma))

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
            #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	    #pr 'num',bottom[0].num

        num = bottom[0].num
        self.spatial_dim = bottom[0].height * bottom[0].width
        if (self.spatial_dim != 1):
            raise Exception("self.spatial_dim != 1")

        # processing bottom[0] to softmax probability
        scores = np.array(bottom[0].data)
        tmp = np.tile(np.max(scores,axis=1),4320)
        tmp = tmp.reshape(scores.T.shape).T
        scores -= tmp
        exp_scores = np.exp(scores)
        self.prob_data = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #print 'self.prob_data', self.prob_data

        self.label = np.array(bottom[1].data).reshape(num)
        self.label = self.label.astype(int)

        self.weight = np.zeros_like(self.label)
        self.weight[self.label < 10000] = self.pos_weight
        self.weight[self.label >= 10000] = self.neg_weight
        self.weight = np.tile(self.weight, self.dim).reshape(self.dim, num).T.reshape(num, 12, 360)

        self.label[self.label >= 10000] -= 10000

        if (self.dim < np.max(self.label)):
            raise Exception("self.label value exceeds dimension")


        self.cls_idx = np.array(self.label / (self.period))
        nonbg_ind = (self.cls_idx != 12)
        bg_ind = (self.cls_idx == 12)

        probs_cls_data = self.prob_data.reshape(num, 12, 360)

        probs_cls_sum = np.zeros(shape=num,dtype=np.float)
        probs_cls_sum = np.sum(abs(probs_cls_data[np.arange(num), self.cls_idx]), axis = 1)
        probs_cls_sum = np.tile(probs_cls_sum, 360).reshape(360, num).T
        #print 'probs_cls_sum.shape', probs_cls_sum[nonbg_ind].shape
        #print 'probs_cls_sum', probs_cls_sum

        #print 'probs_cls_data:before', probs_cls_data[0, self.cls_idx[0]]
        probs_cls_data[nonbg_ind, self.cls_idx[nonbg_ind]]/= probs_cls_sum[nonbg_ind]
        #print 'probs_cls_sum[0]',probs_cls_sum[0]
        #print 'probs_cls_data:after', probs_cls_data[0, self.cls_idx[0]]

        diff = np.zeros_like(probs_cls_data)
        diff[nonbg_ind, self.cls_idx[nonbg_ind]] = np.array(probs_cls_data[nonbg_ind, self.cls_idx[nonbg_ind]] * self.weights_sum)
        diff *= self.weight

        # convert to 360-class self.label
        view_label = self.label % (self.period)
        view_label = np.tile(view_label, 2*self.bandwidth+1)
        view_label = view_label.reshape(2*self.bandwidth+1, num).T

        k = np.arange(-self.bandwidth, self.bandwidth+1)
        k = np.tile(k, num)
        k = k.reshape(num, 2*self.bandwidth+1)

        # e.g. view_label+k=-3 --> 357
        view_k = (view_label + k) % self.period


        cls_idx_broad = np.tile (self.cls_idx * self.period, 2*self.bandwidth+1).reshape(2*self.bandwidth+1, num).T
        # convert back to 4320-class self.label
        ex_ind = np.tile(range(num),2*self.bandwidth+1).reshape(2*self.bandwidth+1, num).T *self.dim
        label_value_k = view_k + cls_idx_broad + ex_ind

        self.prob_data = probs_cls_data.reshape(num* self.dim)
        diff = diff.reshape(num* self.dim)
        # self.diff = diff.copy()
        tmp = self.prob_data[label_value_k]
        tmp = tmp[nonbg_ind]

        # loss is weighted by exp(-|dist|/sigma)
        tmp -= np.exp(-abs(k[nonbg_ind]) / self.sigma) * np.log(tmp + 10**(-37))
        tmp = np.sum(tmp, axis=1) * self.weight.T[0][0][nonbg_ind]
        top[0].data[...] = np.sum(tmp) / num
        with open('baseline_all/'+self.type_+'_loss.txt', "a") as f:
            f.write(str(top[0].data[0]))
            f.write('\n')
        f.close()
        #ipdb.set_trace()
        # Equivalent to self.diff[label_value_k][nonbg_ind] = diff[label_value_k][nonbg_ind] - np.exp(-abs(k[nonbg_ind]) / self.sigma) * self.weight.reshape(num* self.dim)[label_value_k][nonbg_ind]
        tmp = diff[label_value_k]
        tmp[nonbg_ind] -= np.exp(-abs(k[nonbg_ind]) / self.sigma) * self.weight.reshape(num* self.dim)[label_value_k][nonbg_ind]
        diff[label_value_k] = tmp
        diff /= num
        self.diff = diff.reshape(*bottom[0].data.shape)

    def backward(self, top, propagate_down, bottom):

        #for i in range(2):
        if propagate_down[1]:
            raise Exception("Layer cannot backprop to self.label inputs.")
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff / bottom[0].num

class SoftmaxViewLoss_active(caffe.Layer):

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

        params = eval(self.param_str_)
        self.type_ = params['type']
        self.bandwidth = params.get('self.bandwidth', 5)
        self.sigma = float(params.get('sigma', 3))
        self.pos_weight=params.get('pos_weight', 1)
        self.neg_weight=params.get('neg_weight', 0)
        self.period=int(params.get('period', 360))

        self.weights_sum = np.sum(np.exp(-abs(np.arange(-self.bandwidth, self.bandwidth+1))/ self.sigma))

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
            #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	    #pr 'num',bottom[0].num

        num = bottom[0].num
        self.spatial_dim = bottom[0].height * bottom[0].width
        if (self.spatial_dim != 1):
            raise Exception("self.spatial_dim != 1")

        # processing bottom[0] to softmax probability
        scores = np.array(bottom[0].data)
        tmp = np.tile(np.max(scores,axis=1),4320)
        tmp = tmp.reshape(scores.T.shape).T
        scores -= tmp
        exp_scores = np.exp(scores)
        self.prob_data = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #print 'self.prob_data', self.prob_data

        self.label = np.array(bottom[1].data).reshape(num)
        self.label = self.label.astype(int)

        self.weight = np.zeros_like(self.label)
        self.weight[self.label < 10000] = self.pos_weight
        self.weight[self.label >= 10000] = self.neg_weight
        self.weight = np.tile(self.weight, self.dim).reshape(self.dim, num).T.reshape(num, 12, 360)

        self.label[self.label >= 10000] -= 10000

        if (self.dim <= np.max(self.label)):
            raise Exception("self.label value exceeds dimension")


        self.cls_idx = np.array(self.label / (self.period))
        nonbg_ind = (self.cls_idx != 12)

        probs_cls_data = self.prob_data.reshape(num, 12, 360)

        probs_cls_sum = np.zeros(shape=num,dtype=np.float)
        probs_cls_sum[nonbg_ind] = np.sum(abs(probs_cls_data[nonbg_ind, self.cls_idx[nonbg_ind]]), axis = 1)
        probs_cls_sum = np.tile(probs_cls_sum, 360).reshape(360, num).T
        #print 'probs_cls_sum.shape', probs_cls_sum[nonbg_ind].shape
        #print 'probs_cls_sum', probs_cls_sum

        #print 'probs_cls_data:before', probs_cls_data[0, self.cls_idx[0]]
        probs_cls_data[nonbg_ind, self.cls_idx[nonbg_ind]] /= probs_cls_sum[nonbg_ind]
        #print 'probs_cls_sum[0]',probs_cls_sum[0]
        #print 'probs_cls_data:after', probs_cls_data[0, self.cls_idx[0]]

        diff = np.zeros_like(probs_cls_data)
        diff[nonbg_ind, self.cls_idx[nonbg_ind]] = np.array(probs_cls_data[nonbg_ind, self.cls_idx[nonbg_ind]] * self.weights_sum)
        diff *= self.weight

        # convert to 360-class self.label
        view_label = self.label % (self.period)
        view_label = np.tile(view_label, 2*self.bandwidth+1)
        view_label = view_label.reshape(2*self.bandwidth+1, num).T

        k = np.arange(-self.bandwidth, self.bandwidth+1)
        k = np.tile(k, num)
        k = k.reshape(num, 2*self.bandwidth+1)

        # e.g. view_label+k=-3 --> 357
        view_k = (view_label + k) % self.period


        cls_idx_broad = np.tile (self.cls_idx * self.period, 2*self.bandwidth+1).reshape(2*self.bandwidth+1, num).T
        # convert back to 4320-class self.label
        ex_ind = np.tile(range(num),2*self.bandwidth+1).reshape(2*self.bandwidth+1, num).T *self.dim
        label_value_k = view_k + cls_idx_broad + ex_ind

        self.prob_data = probs_cls_data.reshape(num* self.dim)
        diff = diff.reshape(num* self.dim)
        # self.diff = diff.copy()
        tmp = self.prob_data[label_value_k]
        tmp = tmp[nonbg_ind]

        # loss is weighted by exp(-|dist|/sigma)
        tmp -= np.exp(-abs(k[nonbg_ind]) / self.sigma) * np.log(tmp + 10**(-37))
        tmp = np.sum(tmp, axis=1) * self.weight.T[0][0][nonbg_ind]
        top[0].data[...] = np.sum(tmp) / num
        with open('ohem_all/'+self.type_+'_loss.txt', "a") as f:
            f.write(str(top[0].data[0]))
            f.write('\n')
        f.close()

        #ipdb.set_trace()
        # Equivalent to self.diff[label_value_k][nonbg_ind] = diff[label_value_k][nonbg_ind] - np.exp(-abs(k[nonbg_ind]) / self.sigma) * self.weight.reshape(num* self.dim)[label_value_k][nonbg_ind]
        tmp = diff[label_value_k]
        tmp[nonbg_ind] -= np.exp(-abs(k[nonbg_ind]) / self.sigma) * self.weight.reshape(num* self.dim)[label_value_k][nonbg_ind]
        diff[label_value_k] = tmp
        diff /= num
        self.diff = diff.reshape(*bottom[0].data.shape)

    def backward(self, top, propagate_down, bottom):

        #for i in range(2):
        if propagate_down[1]:
            raise Exception("Layer cannot backprop to self.label inputs.")
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff / bottom[0].num
