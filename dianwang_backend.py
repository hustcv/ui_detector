#!/usr/bin/env python

# --------------------------------------------------------
# Backend of dianwang detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Duino Du
# --------------------------------------------------------


import _init_paths_tf
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from networks.factory import get_network
#from fast_rcnn.nms_wrapper import nms
from nms.py_cpu_nms import py_cpu_nms as nms
import pprint
import tensorflow as tf
import time, os, sys, cv2
from utils.timer import Timer
from fast_rcnn.test import im_detect

import numpy as np
import pickle

import detect_mser as textreader

color = {"tower":(237, 27, 37), "insulator":(254, 174, 200), "hammer":(0, 162, 232), "nest":(253, 127, 39), "text":(243, 234, 71)}
#color = {"tower":(37, 27, 237), "insulator":(200, 174, 254), "hammer":(232, 162, 0), "nest":(39, 127, 253), "text":(71, 234, 243)}
cls_map= {"1": "nest", "2":"text", "3":"tower","4":"text", "5":"insulator", "6":"hammer", "7":"text"}

class Detector(object):
    """Backend for dianwang"""

    def __init__(self):

        model_file = os.path.join(cfg.ROOT_DIR, "output/faster_rcnn_end2end/final_2017_trainval/VGGnet_fast_rcnn_iter_70000.ckpt")
        num_classes = 8

        config = tf.ConfigProto(allow_soft_placement=True)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=config)
        self.net, _ok = self._init_model(model_file, num_classes) 
        if not _ok:
            print("Error when init model")
            return

        self.blockWidth = 1000
        self.blockHeight = 1000
        self.stride = 500
        self.NMS_THRESH = 0.3

    def detect(self, imgPath, isblock=False):
        im = cv2.imread(imgPath)
        if im.shape[1] > 2000:
            f = im.shape[1] / 1200.
        else:
            f = 1
        #print(("factor:", f))
        im_resize = cv2.resize(im, (int(im.shape[1]/f), int(im.shape[0]/f)))
        timer = Timer()
        timer.tic()
        
        # tower, insulator, hammer
        all_dets = self._detect(im_resize)
        all_dets[:,0] *= f
        all_dets[:,1] *= f
        all_dets[:,2] *= f
        all_dets[:,3] *= f

        result = {"tower":0, "insulator":0, "hammer":0, "nest":0, "text":0, "bad_insulator":0, "bad_hammer":0, "textStr":"null"}

        #print(("first:",all_dets.shape), all_dets[:,-1])

        ## text, nest
        #for roi in all_dets:
        #    if roi[5] == 3: # tower id
        #        roi = roi.astype(np.int)
        #        tower_roi = im[roi[1]:roi[3], roi[0]:roi[2]]
        #        dets = self._block_im_detect(tower_roi)
        #        # edit coordinate
        #        dets[:,0] += roi[0]
        #        dets[:,1] += roi[1]
        #        dets[:,2] += roi[0]
        #        dets[:,3] += roi[1]
        #        
        #        for i in [1,2,6,7]:
        #            inds = np.where(dets[:, -1] == i)[0]
        #            dets = dets[inds, :]
        #            all_dets = np.vstack((all_dets, dets)) 

        #print(("second:",all_dets.shape), all_dets[:,-1])
                
        keep = nms(all_dets, self.NMS_THRESH)
        all_dets = all_dets[keep, :]

        timer.toc()
        #print((('Detection took {:.3f}s for {}').format(timer.total_time, imgPath)))
    
        #tpnum = None
        #tpindx = None
        #for i in range(all_dets.shape[0]):
        #    class_name = cls_map[str(int(all_dets[i, -1]))]
        #    result[class_name] += 1

        #    # TODO: text recognition
        #    if int(all_dets[i, -1]) == 2 and tpnum is None:
        #        roi = all_dets[i].astype(np.int)
        #        tpimg = im[roi[1]:roi[3],roi[0]:roi[2]].copy()

        #        tpnum = textreader.getTPNumber(tpimg)

        #        if tpnum is not None:
        #            tpindx = i

        #        #cv2.imshow('',tpimg)
        #        #cv2.waitKey()

        #if tpnum is not None:
        #    result['textStr'] = tpnum
        #    result['text'] = 1
        #else:
        #    result['text'] = 0

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #return im, all_dets, result, tpindx
        return im, all_dets, result

    def _init_model(self, model, num_classes, net_name='VGGnet_test'):
    
        net = get_network(net_name, num_classes)

        if not os.path.exists(model+'.meta'):
            print(("{} not exist".format(model)))
            return net, False
        
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = 0
        
        # load weights
        print((('Loading model weights from {:s}').format(model)))
        saver = tf.train.Saver()
        saver.restore(self.sess, model)
        print('Finish')
    
        # Warmup on a dummy image
        im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
        for i in range(2):
            _, _, _= im_detect(self.sess, net, im)
        return net, True

    def _detect(self, im, score_thresh=0.8):
        """
        meta-detect
        """
        scores, boxes, _ = im_detect(self.sess, self.net, im)

        # Visualize detections for each class
        all_dets = np.zeros(shape=(0,6), dtype=np.float32)
        for cls_ind in range(scores.shape[1]-1):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            cls_inds = np.zeros((cls_scores.shape[0]))
            cls_inds.fill(cls_ind)
    
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis],
                              cls_inds[:, np.newaxis])).astype(np.float32)

            inds = np.where(dets[:, -2] >= score_thresh)[0]
            dets = dets[inds, :]
            keep = nms(dets, self.NMS_THRESH)
            dets = dets[keep, :]
            all_dets = np.vstack((all_dets, dets)) 
        return all_dets

    def _block_im_detect(self, im):
    
        height, width = im.shape[:2]
    
        # create block axis
        if width > self.blockWidth: 
            x_ind = [x*self.stride for x in range((width-self.blockWidth)/self.stride+1)]
            x_ind.append(width-self.blockWidth)
        else:
            x_ind = [0]
        if height > self.blockHeight: 
            y_ind = [x*self.stride for x in range((height-self.blockHeight)/self.stride+1)]
            y_ind.append(height-self.blockHeight)
        else:
            y_ind = [0]
        
        # detect each block
        all_dets = np.zeros(shape=(0,6), dtype=np.float32)
        for y in y_ind:
            for x in x_ind:
                x1 = x
                x2 = min(x+self.blockWidth, width)
                y1 = y
                y2 = min(y+self.blockHeight, height)
                img = im[y1:y2, x1:x2]
                dets = self._detect(img)
                dets[:,0] += x1
                dets[:,1] += y1
                dets[:,2] += x1
                dets[:,3] += y1
                all_dets = np.vstack((all_dets, dets)) 
        
        keep = nms(all_dets, self.NMS_THRESH)
        all_dets = all_dets[keep, :]
    
        return all_dets        

def _debug(dets, im, id):
    bgImg = np.zeros_like(im)
    for i in range(dets.shape[0]):
        bbox = dets[i, :4].astype(np.int)
        cv2.rectangle(bgImg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 1)
    #cv2.resize(bgImg, (bgImg.shape[1]/4, bgImg.shape[0]/4))
    cv2.imwrite('/tmp/debug/{}.jpg'.format(id), bgImg)



if __name__ == "__main__":
    from fileIO import write2xml
    net = Detector()
    root = '/home/cv/data/VOCdevkit/DIAN2017/JPEGImages'
    annoDir = root + '_anno'

    if not os.path.exists(annoDir):
        os.makedirs(annoDir)
    imgfiles = sorted([os.path.join(root, x) for x in sorted(os.listdir(root)) if x.endswith('.jpg')])

    fetch = lambda imgfile: os.path.basename(imgfile)[:-4]+'.xml'
    for imgfile in imgfiles[7984:]:
        img_ret, all_dets, data_ret = net.detect(imgfile)
        cls_names = []
        height, width,_ = cv2.imread(imgfile).shape
        for i in range(all_dets.shape[0]): # [x1, y1, x2, y2, score, cls_ind]
            class_name = cls_map[str(int(all_dets[i, -1]))]
            cls_names.append(class_name)
        filename = os.path.join(annoDir, fetch(imgfile))
        write2xml(filename, height, width, cls_names, all_dets[:,:4].astype(int))
        print(filename)
