import numpy as np
import caffe



class MySoftmaxLayer(caffe.Layer):

    def setup(self, bottom, top):

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
	    #params = eval(self.param_str_)
        #self.split = params['split']

    def reshape(self, bottom, top):

        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
            #raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	#print 'num',bottom[0].num
        scores = np.array(bottom[0].data)
        #print 'bottom',bottom[0].data.shape,bottom[0].data
        tmp = np.tile(np.max(scores,axis=1),np.max(bottom[1].data).astype(int)+1)
        tmp = tmp.reshape(scores.T.shape).T
        #print 'tmp',tmp.shape,tmp
        #print 'scores',scores.shape,scores
        scores = scores-tmp
        exp_scores = np.exp(scores)
        #64*10
        #tmp = np.sum(exp_scores, axis=1, keepdims=True)
        #tmp = np.tile(tmp,10).reshape(scores.T.shape).T
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(bottom[0].num),np.array(bottom[1].data,dtype=np.uint16).reshape(bottom[1].num)]+10**(-10))
        #print 'correct',probs[range(bottom[0].num),np.array(bottom[1].data,dtype=np.uint16).reshape(bottom[1].num)]+10**(-10)
        #print 'log',correct_logprobs
        data_loss = np.sum(correct_logprobs)/bottom[0].num
        '''
        if self.split is 'training':
            if share_data.flag:
                share_data.data_loss = []
            share_data.data_loss = np.append(share_data.data_loss, correct_logprobs)
	    #print 'loss', share_data.data_loss.shape
        '''

        self.diff[...] = probs
        top[0].data[...] = data_loss


    def backward(self, top, propagate_down, bottom):

        delta = self.diff


        #for i in range(2):
        if propagate_down[1]:
            raise Exception("Layer cannot backprop to label inputs.")
        if propagate_down[0]:
            delta[range(bottom[0].num), np.array(bottom[1].data,dtype=np.uint16).reshape(bottom[1].num)] -= 1
            bottom[0].diff[...] = delta/bottom[0].num
