import numpy as np
import caffe
import share_data as sd
class SoftmaxViewLoss_sub(caffe.Layer):

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

        params = eval(self.param_str_)
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
            #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        # loss output is scalar
        top[0].reshape(1)
	# top[1].reshape(1)
	# print bottom[0].data.shape
        top[1].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        if (self.iter % 100 ==0):
             if 'az' in self.type_:
                 sd.loss_az = []
             if 'el' in self.type_:
                 sd.loss_el = []
             if 't' in self.type_:
                 sd.loss_t = []
        num = bottom[0].num
        #print 'bandwidth',self.bandwidth
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
        for i in range(num):
             loss_i = np.float32(0)
             for j in range(self.spatial_dim):
                 label_value = self.label[i * self.spatial_dim + j]
                 if (label_value < 10000):
                     weight = self.pos_weight
                 else :
                     weight = self.neg_weight
                     label_value = label_value - 10000
		 #print 'label_value',label_value
		 #print 'weight', weight


          # Added by rqi, full of HARD CODING..
          # ASSUMPTION: classes number is 12*360 or 12*360 + 1
                 cls_idx = int(label_value / self.period )# 0~11,12->bkg
                 if (cls_idx == 12) : continue # no loss for bkg

          # normalize prob_data of sample i to probs (360 numbers)
          # inside the category corresponding to label_value
		 #print 'prob_data1',self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)]
                 probs_cls_sum = np.float32(np.sum(self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)]))
                 self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)] /= probs_cls_sum
		 #for m in range(800,841):
	    	    # print self.prob_data[m]
		 #print 'probs_cls_sum', probs_cls_sum
		 #print 'prob_data2',self.prob_data[range(i*dim + cls_idx*self.period,i*dim + cls_idx*self.period+self.period)]


                # convert to 360-class label
                 view_label = label_value % self.period
                 tmp_loss = np.float32(0)
                 for k in range(-self.bandwidth, self.bandwidth+1):
                    # get positive modulo
                    # e.g. view_label+k=-3 --> 357
                     view_k = (view_label + k) % self.period
                    # convert back to 4320-class label
                     label_value_k = view_k + cls_idx * self.period
		     #print 'label_value_k',label_value_k
                    # loss is weighted by exp(-|dist|/sigma)
		     #print 'exp', np.exp(-abs(k)/self.sigma) * np.log(max(self.prob_data[i * dim + label_value_k * self.spatial_dim + j],10**(-37)))
                     tmp_loss -= np.exp(-abs(k)/self.sigma) * np.log(max(self.prob_data[i * dim + label_value_k * self.spatial_dim + j],10**(-37)))
		     #print 'tmp_loss',tmp_loss

                # scale loss with loss weight
                 loss_i += tmp_loss * weight
                 loss += tmp_loss * weight
             if 'az' in self.type_:
                 sd.loss_az = np.append(sd.loss_az,loss_i )
             if 'el' in self.type_:
                 sd.loss_el = np.append(sd.loss_el,loss_i )
             if 't' in self.type_:
                 sd.loss_t = np.append(sd.loss_t,loss_i )
             #self.ind = np.append(self.ind, self.label[i])
	#if self.iter == 2098:
            #np.save(self.type_+'_loss',self.loss)
            #np.save(self.type_+'_ang',self.ind)
          #loss -= log(std::max(prob_data[i * dim +
          #    label_value * spatial_dim + j],Dtype(FLT_MIN)))

        top[0].data[...] = loss / num / self.spatial_dim
        top[1].data[...] = self.prob_data.reshape(num,dim)
        self.iter += 1


    def backward(self, top, propagate_down, bottom):
        self.iter -= 1
        num = bottom[0].num
        dim = 4320
        #print 'flag',sd.flag
        #print self.mine
        #for i in range(2):
        if propagate_down[1]:
            raise Exception("Layer cannot backprop to self.label inputs.")
        if propagate_down[0]:
            if  (self.iter % 200 < 100):
                if (self.iter % 50 == 0) and 'az' in self.type_:
                    print 'no backprop at %d' %self.iter
                bottom[0].diff[...] = np.zeros_like(bottom[0].data)
            else:
                diff = np.zeros_like(self.prob_data)
                for i in range(num):
                    if (self.iter >= 24000) and sd.flag[i]:
                        continue
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
                        if (cls_idx == 12) : continue # no loss for bkg
                        self.cnt += 1
                        if 'az' in self.type_:
                            self.mine = np.append(self.mine,i)
                        diff[range(i*dim + cls_idx*self.period, i*dim + cls_idx*self.period + self.period)] = self.prob_data[range(i*dim + cls_idx*self.period, i*dim + cls_idx*self.period + self.period)] * self.weights_sum
                        diff[range(i*dim,i*dim+dim)] *= weight
                        view_label = label_value % self.period
                        for k in range(-self.bandwidth, self.bandwidth+1):
                            # get positive modulo
                            # e.g. view_label+k=-3 --> 357
                            view_k = (view_label + k) % self.period
                            # convert back to 4320-class label
                            label_value_k = view_k + cls_idx * self.period
                            diff[i * dim + label_value_k * self.spatial_dim + j] -= np.exp(-abs(k)/self.sigma) * weight
                diff = diff.reshape(bottom[0].data.shape)
                #print self.type_

                self.count = np.append(self.count, self.cnt)
                #print self.count
                bottom[0].diff[...] = diff /(self.count[-1]-self.count[-2]) / self.spatial_dim


                if (self.iter >= 24000) and ('az' in self.type_):
                    np.save('ohem_con/sub/mine_loss'+str(self.iter)+'.npy',self.mine)
                    if (self.iter % 2000 == 100):
                        np.save('ohem_con/sub/count'+str(self.iter)+'.npy',self.count[np.arange(1,len(self.count))])
                        self.count = [self.count[-1]]
                    self.mine = []
        self.iter += 1
