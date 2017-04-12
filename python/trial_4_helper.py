import sys, os
from os import listdir
import tempfile
sys.path.append('/home/qchen/caffe/python')
import caffe
from lmdb_para import *
sys.path.append('/mnt/4T-HD/qchen/RenderForCNN_4')
sys.path.append(os.path.join('/mnt/4T-HD/qchen/RenderForCNN_4', 'render_pipeline'))
cache_dir = '/mnt/4T-HD/qchen/cache'
log_dir = 'experiment/trial_4'
import matplotlib.pyplot as plt
import time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
from global_variables import *
import random
import re
import scipy.misc
import pdb
train_image_folder = 'data/mine/syn_lmdb_train_image_rand'
train_label_folder = 'data/mine/syn_lmdb_train_label_rand'

model_class = np.load('statistics/model_class.npy')
names = ['aeroplane.txt','bicycle.txt','boat.txt','bottle.txt','bus.txt','car.txt','chair.txt','diningtable.txt','motorbike.txt','sofa.txt','train.txt','tvmonitor.txt']
kernels = np.load('statistics/distances_kernels.npy')
g_dist= np.load('statistics/init_dist.npy')
tilt_dist = np.load('statistics/tilt_dist.npy')
def update_dist(acc, val_count):
    dist = np.empty_like(acc)
    for i in range(len(acc)):
        if val_count[i] == 0:
            dist[i] = 0
        else: dist[i] = acc[i] / val_count[i]
    tmp = g_dist * np.exp(-2 * dist)
    return tmp / np.sum(tmp)

def prefetch(ind, np_folder, i):
    if not os.path.exists(np_folder):
            os.makedirs(np_folder)
     
    ind_new = np.where(ind < 0)[0]
    ind_old = np.where(ind >= 0)[0]
    if len(ind_new) :
        ind_new = ind[ind_new]
        ind_new = -ind_new - 1
        data_new = np.load(os.path.join(np_folder, 'image.npy'))
        data_new = data_new[ind_new]
        label_new = np.load(os.path.join(np_folder, 'label.npy'))
        label_new = label_new[ind_new]


    if len(ind_old):
        ind_old = ind[ind_old]
        data_old = read_lmdb(train_image_folder, ind_old)
        data_old = np.asarray(data_old)
        data_old = data_old.reshape((len(ind_old), 3, 227, 227))
        data_old = data_old.astype(np.uint8)
    
        label_old = read_lmdb(train_label_folder, ind_old)
        label_old = np.asarray(label_old)
        label_old = label_old.reshape((len(ind_old), 4, 1, 1))
        label_old = label_old.astype(np.uint16)

    if (len(ind_new) and len(ind_old)):
        data = np.row_stack((data_new, data_old))
        label = np.row_stack((label_new, label_old))
    elif len(ind_new):
        data = np.asarray(data_new)
        label = np.asarray(label_new)
    elif len(ind_old):
        data = np.asarray(data_old)
        label = np.asarray(label_old)
    #pdb.set_trace()
    data = data.reshape(len(ind), 3, 227,227)
    label = label.reshape(len(ind), 4, 1,1)
    np.save(os.path.join(np_folder,str(i)+'_image.npy'), data)
    np.save(os.path.join(np_folder,str(i)+'_label.npy'), label)

def sample_views(dist, size, dirname, epoch):
    log = os.path.join(dirname, 'sample.log')
    if not os.path.exists(dirname):
            os.makedirs(dirname)
    f = open(log, 'w')
    sample_views = np.array([])
    new_views_ind = np.array([])
    az_els = np.random.choice(12*12*36, size, p = dist)
    unique, counts = np.unique(az_els, return_counts=True)
    tmp = dict(zip(unique, counts))
    for i in tmp.keys():
        file = os.path.join('statistics/dist_4',str(i)+'.npy')
        if (os.path.exists(file)):
            inds = np.load(file)
            if tmp[i] >= len(inds):
                f.write('%d samples from %s used up\n' %(len(inds),file))
                sample_views = np.append(sample_views, inds)
                new_inds = np.tile([-1],tmp[i]-len(inds))
                sample_views = np.append(sample_views, new_inds)
                new_views_ind = np.append(new_views_ind, new_inds *(-i)).astype(int)
                os.system('rm %s &' %file)
            else:
                sample_views = np.append(sample_views, inds[np.arange(tmp[i])])
                inds = np.delete(inds, np.arange(tmp[i]))
                f.write('get %d samples from %s\n' %(tmp[i], file))
                f.write('remain %d samples in %s\n' %(len(inds), file))
                np.save(file, inds)
        else:
            f.write('trying to sample from %s, but no instances\n' %file)
            inds = np.tile([-1],tmp[i])
            sample_views = np.append(sample_views, inds)
            new_views_ind = (np.append(new_views_ind, inds *(-i))).astype(int)
    render_num = len(new_views_ind)
    f.write('sample %d images in total\n' %len(sample_views))
    start_time = time.time()
    if render_num > 0:
        new_views = generate_views(new_views_ind)
        render_one_epoch(new_views, dirname, epoch)
        while (render_num > len(os.listdir(os.path.join(g_syn_images_folder, str(epoch))))):
            diff = render_num - len(os.listdir(os.path.join(g_syn_images_folder, str(epoch))))
            print ('--- still rendering %d images ---' %diff)
            rest_ind = np.random.choice(new_views_ind, diff)
            view_params = generate_views(rest_ind)
            render_one_epoch(view_params, dirname, epoch)
        print 'render %d images takes %d seconds' %(render_num, (time.time()-start_time))
        crop_overlay_epoch(epoch)
    np_folder = os.path.join(g_syn_images_lmdb_folder,str(epoch))
    sys.stdout.flush()
    np.random.shuffle(sample_views)
    sample_views = sample_views.astype(int)
    ind = np.where(sample_views == -1)[0]
    sample_views[ind]= -np.arange(1,len(ind)+1)
    length = len(sample_views)
    sample_views = np.split(sample_views, 1000)
    for i in range(1000):
        tmp = sample_views[i].reshape(length/1000)
        prefetch(np.sort(tmp), np_folder, i)


def generate_views(az_els):
    sample_views = np.array([])
    size = len(az_els)
    azs = np.random.randint(30, size = size) + np.random.rand(size)
    els = np.random.randint(10, size = size) + np.random.rand(size)
    tlts = np.random.randint(10, size = size) + np.random.rand(size)
    ind = 0
    clss = az_els / (12*36)
    # for cls in clss:
    #     print cls, len(tilt_dist[cls*36: (cls*36+36)])
    tlts_base = [np.random.choice(36, p = tilt_dist[cls*36: (cls*36+36)]) * 10 for cls in clss]
    tlts += tlts_base
    els = (az_els % 36) * 10 + els
    tmp = az_els - clss*12*36
    azs = (tmp / 36) * 30 + azs
    models = [np.random.randint(model_class[i],model_class[i+1],5) for i in range(12)]
    model_idx = [np.random.choice(models[cls]) for cls in clss]
    sample_views = np.column_stack((model_idx, clss, azs, els, tlts))
    idx = np.argsort(model_idx)
    sample_views = sample_views[idx]
    classes = sample_views[:,1]
    distances = np.empty_like(azs)
    for i in range(12):
        idx = np.where(classes == i)[0]
        num = len(idx)
        kernel = kernels[i]
        new_sample = kernel.resample(num)
        new_sample = np.maximum(new_sample, 1)
        new_sample = np.minimum(new_sample, 29)
        distances[idx] = new_sample
    sample_views = np.column_stack((sample_views, distances))
    return sample_views


def render_one_epoch(view_params, dirname, epoch):
    model_idx = view_params[:,0]
    model = model_idx[0]
    f = open(os.path.join(dirname, 'commands.txt'),'w')
    count = 0
    tmp = open(os.path.join(dirname,'0'),'w')
    for i in range(view_params.shape[0]):
        if model_idx[i] != model:
            tmp.close()
            command = '%s %s --background --python %s -- %s %s >> %s\n' % (g_blender_executable_path, g_blank_blend_file_path, 'python/render_by_para.py', tmp.name, os.path.join(g_syn_images_folder, str(epoch)), os.path.join(dirname, 'blender.log'))
            tmp = open(os.path.join(dirname,str(i)),'w')
            f.write(command)
            count += 1
            model = model_idx[i]
        tmp_string = '%f %f %f %f %f %f\n' % (model_idx[i], view_params[i][1], view_params[i][2], view_params[i][3], view_params[i][4], view_params[i][5])
        tmp.write(tmp_string)


    tmp.close()
    command = '%s %s --background --python %s -- %s %s >> %s\n' % (g_blender_executable_path, g_blank_blend_file_path, 'python/render_by_para.py', tmp.name, os.path.join(g_syn_images_folder, str(epoch)), os.path.join(dirname, 'blender.log'))
    f.write(command)
    f.close()
    print(command)
    os.system('parallel < %s' %os.path.join(dirname, 'commands.txt'))


#view_params = sample_views(g_dist,300)
#render_one_epoch(view_params, '/home/qchen/workspace/py-faster-rcnn/caffe-fast-rcnn/experiment/trial_1', 0)


def process_img(im):
    #im = im.resize((g_images_resize_dim, g_images_resize_dim, 3), Image.ANTIALIAS)
    # convert to array
    #im = scipy.misc.imresize(im, (g_images_resize_dim, g_images_resize_dim), interp='bilinear')
    im = Image.fromarray(im, 'RGB')
    im = im.resize((g_images_resize_dim, g_images_resize_dim), Image.ANTIALIAS)
    im = np.array(im)
    # convert gray to color
    if len(np.shape(im)) == 2: # gray image
        im = im[:,:,np.newaxis]
        im = np.tile(im, [1,1,3])
    # change RGB to BGR
    im = im[:,:,::-1]
    # change H*W*C to C*H*W
    im = im.transpose((2,0,1))
    return im

def crop_gray(I, bgColor, truncationParam):
    [nr, nc] = I.shape
    colsum = (np.sum(I == bgColor, axis=0) != nr)
    colsum = colsum.T.ravel()
    #print np.nonzero(colsum)[0]
    rowsum = (np.sum(I == bgColor, axis=1) != nc)
    rowsum = rowsum.T.ravel()
    left = np.nonzero(colsum)[0][0]
    right = np.nonzero(colsum)[0][-1]
    top = np.nonzero(rowsum)[0][0]
    bottom = np.nonzero(rowsum)[0][-1]
    width = right - left + 1
    height = bottom - top + 1
    # strecth
    dx1 = width * truncationParam[0]# left
    dx2 = width * truncationParam[1]# right
    dy1 = height * truncationParam[2]# top
    dy2 = height * truncationParam[3]# bottom

    leftnew = max([0, left + dx1])
    leftnew = min([leftnew, nc - 1])
    rightnew = max([0, right + dx2])
    rightnew = min([rightnew, nc - 1])
    if leftnew > rightnew:
        leftnew = left
        rightnew = right

    topnew = max([0, top + dy1])
    topnew = min([topnew, nr - 1])
    bottomnew = max([0, bottom + dy2])
    bottomnew = min([bottomnew, nr - 1])
    if topnew > bottomnew:
        topnew = top
        bottomnew = bottom
    left = int(round(leftnew))
    right = int(round(rightnew))
    top = int(round(topnew))
    bottom = int(round(bottomnew))
    #print 'here', top, bottom, left, right
    I = I[top:(bottom+1), left:(right+1)]    
    return I,top,bottom,left,right

def crop_image(name, src_folder, image_file, src_image_file):

    truncation_distr_file = os.path.join(g_truncation_distribution_folder,name)
    lines = [line.rstrip() for line in open(truncation_distr_file, 'r')]
    ll = lines[np.random.randint(len(lines))]
    ll = ll.split(' ')
    truncationParametersSub = np.array([float(x) for x in ll])
    try:
        im = Image.open(src_image_file)
        rgba=np.array(im)
        I = rgba[:,:,0:3]
        alpha = rgba[:,:,3]
    except RuntimeError:
        print 'Failed to read %s\n' %src_image_file

    alpha, top, bottom, left, right = crop_gray(alpha, 0, truncationParametersSub)
    I = I[top:(bottom+1),left:(right+1)]
    if (I.size == 0):
        print 'Failed to crop %s (empty image after crop)\n' %src_image_file
    else:
        #pilImage = Image.frombuffer('RGBA',I.shape[0:2],img,'raw','RGBA',0,1)
        #pilImage.save(os.path.join(g_syn_images_cropped_folder, image_file))
        return I, alpha

def overlay_image(I, alpha, image_file, src_image_file, dst_folder, bkgFilelist, bkgFolder, clutteredBkgRatio):
    sunImageList = [line.rstrip() for line in open(bkgFilelist, 'r')]
    fh, fw, _ = I.shape
    mask = alpha.astype(float) / 255
    mask=np.tile(mask,[3,1,1])
    mask=np.transpose(mask,(1,2,0))
    if random.uniform(0,1) > clutteredBkgRatio:
        I = I.astype(float) * mask + float(random.uniform(0,1)*255) * (1 - mask)
        I = I.astype('uint8')
    else:
        while True:
            ind = np.random.randint(len(sunImageList))
            bg = np.array(Image.open(os.path.join(bkgFolder, sunImageList[ind])))
            bh = bg.shape[0]
            bw = bg.shape[1]
            if (bh < fh) or (bw < fw):
                #print '.'
                continue
            if len(bg.shape)<3: continue
            break
        if (bh == fh): by = 0
        else: by = np.random.randint(bh - fh)
        if (bw == fw): bx = 0
        else: bx = np.random.randint(bw - fw)
        bgcrop = bg[by:(by+fh),bx:(bx+fw)]

        I = I.astype(float) * mask + bgcrop.astype(float) * (1 - mask)
        #I = bgcrop.astype(float) #* (1 - mask)
        I = I.astype('uint8')

    if I.size == 0:
        print 'Failed to overlay %s (empty image after crop)\n' %src_image_file
    else:
        im = Image.fromarray(I)
        im.save(os.path.join(dst_folder, image_file))
        #I = I.transpose((2,0,1))
        return I
def crop_overlay_image(model_class, src_folder, dst_folder, image_file):
    str = re.findall("[-+]?\d+\d*[eE]?[-+]?\d*",image_file)
    label = [int(s) for s in str]
    #idx = np.where(model_class > label[0])[0] - 1
    #idx = idx[0]
    #label.insert(0,idx)
    #print [re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", s) for s in image_file.split('_')]
    label = label[1:5]
    idx = label[0]
    src_image_file = os.path.join(src_folder, image_file)
    I, alpha = crop_image(names[idx], src_folder, image_file, src_image_file)
    I = overlay_image(I, alpha, image_file, src_image_file, dst_folder, g_syn_bkg_filelist, g_syn_bkg_folder, g_syn_cluttered_bkg_ratio)
    I = process_img(I)
    return I, label

class crop_overlay_one_batch(object):

  def __init__(self, model_class,src_folder, dst_folder):
    self.src_folder = src_folder
    self.dst_folder = dst_folder
    self.model_class = model_class
  def __call__(self, image_file):
    return crop_overlay_image(self.model_class, self.src_folder, self.dst_folder, image_file)

def crop_overlay_epoch(epoch):
    src_folder = os.path.join(g_syn_images_folder, str(epoch))
    dst_folder = os.path.join(g_syn_images_bkg_overlaid_folder, str(epoch))
    np_folder = os.path.join(g_syn_images_lmdb_folder,str(epoch))
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    if not os.path.exists(np_folder):
        os.makedirs(np_folder)
    image_files = np.asarray(listdir(src_folder))
    image_num = len(image_files)
    Is = []
    labels = []
    print '%d images in total.' %image_num
    if image_num == 0:
        return
    # for i in range(len(image_files)):
    #     #print i, image_files[i]
    #     I, label = crop_overlay_image(model_class, src_folder, dst_folder, image_files[i])
    #     #print I.shape
    #     Is.append(I)
    #     labels.append(label)
    pool = ThreadPool(12)
    crop_overlay_para = crop_overlay_one_batch(model_class, src_folder, dst_folder)
    print('--- Start croping and overlaying at %s ---' %dst_folder)
    start_time = time.time()
    # for image_file in image_files:
    #     I, label = crop_overlay_image(model_class, src_folder, dst_folder, image_file)
    #     Is.append(I)
    #     labels.append(label)
    for I, label in pool.map(crop_overlay_para, image_files):
        Is.append(I)
        labels.append(label)
    pool.close()
    pool.join()
    print('--- croping and overlaying takes %d seconds ---' %(time.time() - start_time))
    Is = np.asarray(Is)
    labels = np.asarray(labels)
    idx = np.arange(image_num)
    np.random.shuffle(idx)
    new_Is = Is[idx].reshape((image_num, 3, 227, 227))
    new_labels = labels[idx].reshape((image_num, 4, 1, 1))
    np.save(os.path.join(np_folder,'image.npy'), new_Is)
    np.save(os.path.join(np_folder,'label.npy'), new_labels)
   

# start_time = time.time()
# sample_views(g_dist, 5000, os.path.join('experiment/trial_4', str(3)), 3)

# print time.time()-start_time, 's'

def move2cache(epoch):
    path1 =  os.path.join(log_dir, str(epoch))
    path2 = os.path.join(g_syn_images_folder, str(epoch))
    path3 = os.path.join(g_syn_images_bkg_overlaid_folder, str(epoch))
    path4 = os.path.join(g_syn_images_lmdb_folder,str(epoch))
    path21 = os.path.join(cache_dir, 'logs')
    path22 = os.path.join(cache_dir, 'syn_images')
    path23 = os.path.join(cache_dir, 'syn_overlaid')
    path24 = os.path.join(cache_dir, 'syn_lmdbs')
    os.system('mv %s %s &' %(path1, path21))
    os.system('mv %s %s &' %(path2, path22))
    os.system('mv %s %s &' %(path3, path23))
    os.system('mv %s %s &' %(path4, path24))


