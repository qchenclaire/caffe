from compute_acc import *
import caffe
import numpy as np
import share_data as sd
from scipy.stats import entropy
import math

class pas3d_acc_active(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")
        self.iter = 5000

        params = eval(self.param_str_)
        self.period = int(params.get('period', 360))




    def reshape(self, bottom, top):

        if (bottom[0].data.size / bottom[0].num != self.period * 12 ):
            raise Exception("number of classes != 4320.")
        top[0].reshape(1)

    def forward(self, bottom, top):
        sd.flag = np.zeros(bottom[0].num,dtype=np.bool)
        # assign output
        accuracy = float(0)

        p_az = np.array(bottom[0].data)
        p_el = np.array(bottom[1].data)
        p_t = np.array(bottom[2].data)

        az = np.array(bottom[3].data).reshape(bottom[0].num).astype(int)
        el = np.array(bottom[4].data).reshape(bottom[0].num).astype(int)
        t = np.array(bottom[5].data).reshape(bottom[0].num).astype(int)
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
            #tmp = sum(tmp_az * abs(range(360)-az360[i]))
            con_az = np.max(in_az)/np.max(out_az)
            #top_k_az_arg = np.argsort(-tmp_az)[:2]
            #top_k_az = abs(np.sort(-tmp_az)[:2])
            #exp = top_k_az * (abs(top_k_az_arg - az360[i])/15)
            #print exp
            #print sum(exp)
            pred_az = np.argmax(tmp_az)
            #H_az = entropy(in_az)
            #print 'H_az',H_az
            #print 'top_k_az',top_k_az

            tmp_el = p_el[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            in_el = tmp_el[np.arange(el360[i]-5, el360[i] + 6)%360]
            out_el = tmp_el[np.arange(el360[i]+6, 360+el360[i] -5)%360]
            con_el = np.max(in_el)/np.max(out_el)
            pred_el = np.argmax(tmp_el)
            #top_k_el = abs(np.sort(-tmp_el)[:360])
            #H_el = entropy(in_el)
            #print 'H_el',H_el
            #print 'top_k_el',top_k_el

            tmp_t = p_t[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            in_t = tmp_t[np.arange(t360[i]-5, t360[i] + 6)%360]
            out_t = tmp_t[np.arange(t360[i]+6, 360+t360[i] -5)%360]
            con_t = np.max(in_t)/np.max(out_t)
            pred_t = np.argmax(tmp_t)
            #top_k_t = abs(np.sort(-tmp_t)[:360])
            #H_t = entropy(in_t)
            #print 'H_t',H_t
            #print 'top_k_t',top_k_t

            #H = H_az + H_el + H_t
            det = compute_acc([pred_az, pred_el, pred_t],[az360[i], el360[i], t360[i]])
            #print 'det',det
            accuracy += (det < math.pi/6.0)
            if (con_az >=1.2) and (con_el >= 1.2) and (con_t >=1.2) and (det < 0.8 * math.pi/6.0):
                sd.flag[i] = True

            #print 'share_data.mis_col', sd.mis_col

        top[0].data[0] = accuracy / nonbkg_cnt
        self.iter += 1

    def backward(self, top, propagate_down, bottom):
        pass

class pas3d_acc(caffe.Layer):

    def setup(self, bottom, top):

        if bottom[0].num != bottom[1].num:
            raise Exception("The data and label should have the same number.")
        self.iter = 0
        self.acc = 0.0
        self.err = []
        params = eval(self.param_str_)
        self.period = int(params.get('period', 360))

        sd.flag = np.zeros(12*12,dtype=np.float)



    def reshape(self, bottom, top):

        if (bottom[0].data.size / bottom[0].num != self.period * 12 ):
            raise Exception("number of classes != 4320.")
        top[0].reshape(1)

    def forward(self, bottom, top):

        # assign outputi
        accuracy = float(0)
	#err = 0.0

        p_az = np.array(bottom[0].data)
        p_el = np.array(bottom[1].data)
        p_t = np.array(bottom[2].data)

        az = np.array(bottom[3].data).reshape(bottom[0].num).astype(int)
        el = np.array(bottom[4].data).reshape(bottom[0].num).astype(int)
        t = np.array(bottom[5].data).reshape(bottom[0].num).astype(int)
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
            #tmp = sum(tmp_az * abs(range(360)-az360[i]))
            #print 'az', np.max(in_az)/np.max(out_az)
            #top_k_az_arg = np.argsort(-tmp_az)[:2]
            #top_k_az = abs(np.sort(-tmp_az)[:2])
            #exp = top_k_az * (abs(top_k_az_arg - az360[i])/15)
            #print exp
            #print sum(exp)
            pred_az = np.argmax(tmp_az)
            #H_az = entropy(in_az)
            #print 'H_az',H_az
            #print 'top_k_az',top_k_az

            tmp_el = p_el[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            in_el = tmp_el[np.arange(el360[i]-5, el360[i] + 6)%360]
            out_el = tmp_el[np.arange(el360[i]+6, 360+el360[i] -5)%360]
            #print 'el', np.max(in_el)/np.max(out_el)
            pred_el = np.argmax(tmp_el)
            #top_k_el = abs(np.sort(-tmp_el)[:360])
            #H_el = entropy(in_el)
            #print 'H_el',H_el
            #print 'top_k_el',top_k_el

            tmp_t = p_t[i][np.arange(cls_idx[i]*self.period,(cls_idx[i]+1)*self.period)]
            in_t = tmp_t[np.arange(t360[i]-5, t360[i] + 6)%360]
            out_t = tmp_t[np.arange(t360[i]+6, 360+t360[i] -5)%360]
            #print 't', np.max(in_t)/np.max(out_t)
            pred_t = np.argmax(tmp_t)
            #top_k_t = abs(np.sort(-tmp_t)[:360])
            #H_t = entropy(in_t)
            #print 'H_t',H_t
            #print 'top_k_t',top_k_t

            #H = H_az + H_el + H_t
            det = compute_acc([pred_az, pred_el, pred_t],[az360[i], el360[i], t360[i]])
            #print 'det',det
	    self.err = np.append(self.err, det)
            accuracy += (det < math.pi/6.0)

            #print 'share_data.mis_col', sd.mis_col
        self.acc += accuracy / nonbkg_cnt
        #self.err += err / nonbkg_cnt
        top[0].data[0] = accuracy / nonbkg_cnt
        self.iter += 1
        if self.iter == 3912:
            self.acc /= 3912
            self.err = np.median(self.err)
        f1 = open('ohem_err.txt','a')
        f1.write(str(self.err))
        f1.close()
        f2 = open('ohem_acc.txt','a')
        f2.write(str(self.acc))
        f2.close()

    def backward(self, top, propagate_down, bottom):
        pass
