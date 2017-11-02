#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""


import _init_paths_tf
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from networks.factory import get_network
from fast_rcnn.nms_wrapper import nms
import pprint
import tensorflow as tf
import time, os, sys, cv2
from utils.timer import Timer
from fast_rcnn.test import im_detect

import numpy as np
import pickle


#CLASSES = ('__background__', 'tower', 'insulator', 'hammer', 'nest', 'text')
CLASSES = ['_' for x in range(23)]

COLOR = {'tower': (0, 255, 0), 'insulator':(0, 0, 255), 'nest': (255, 0, 255)}

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
sess2 = tf.Session(config=config)

#imdb_name='insulator_2016_test', 
#model_file = "output/faster_rcnn_end2end/result/insulator_2016_trainval_exp1/VGGnet_fast_rcnn_iter_70000.ckpt"

imdb_name='textvoc_2017_test' 
model_file = "output/faster_rcnn_end2end/textvoc_2017_trainval/VGGnet_fast_rcnn_iter_65000.ckpt"

imdb_name='textnest_2017_test' 
model_file = "output/faster_rcnn_end2end/textnest_2017_trainval/VGGnet_fast_rcnn_iter_70000.ckpt"


def init( model= os.path.join(cfg.ROOT_DIR, model_file),
        imdb_name=imdb_name, 
        net_name='VGGnet_test' ):

    global sess
    imdb = get_imdb(imdb_name)
    net = get_network(net_name, imdb.num_classes)
    global CLASSES
    CLASSES = ['_' for x in range(imdb.num_classes)]
    
    if not os.path.exists(model+'.meta'):
        print(("{} not exist".format(model)))
        return
    
    cfg.USE_GPU_NMS = True
    cfg.GPU_ID = 0
    
    # load weights
    saver = tf.train.Saver()
    saver.restore(sess, model)
    print((('Loading model weights from {:s}').format(model)))

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _, _= im_detect(sess, net, im)
    return net


def _detect(sess, net, im, thresh=0.5):
    scores, boxes, _ = im_detect(sess, net, im)
    print((scores.shape))
    print((boxes.shape))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    all_dets = np.zeros(shape=(0,6), dtype=np.float32)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background

        #if cls == 'text':
        #    continue

        #if cls == 'hammer':
        #    continue

        #if cls == 'tower':
        #    continue

        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_inds = np.zeros((cls_scores.shape[0]))
        cls_inds.fill(cls_ind)

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis],
                          cls_inds[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -2] >= thresh)[0]
        dets = dets[inds, :]
        all_dets = np.vstack((all_dets, dets)) 
    return all_dets

def _block_im_detect(sess, net, im):

    blockWidth = 2000
    blockHeight = 2000
    stride = 1800

    height, width = im.shape[:2]

    # create block axis
    if width > blockWidth: 
        x_ind = [x*stride for x in range((width-blockWidth)/stride+1)]
        x_ind.append(width-blockWidth)
    else:
        x_ind = [0]
    if height > blockHeight: 
        y_ind = [x*stride for x in range((height-blockHeight)/stride+1)]
        y_ind.append(height-blockHeight)
    else:
        y_ind = [0]
    
    # detect each block
    all_dets = np.zeros(shape=(0,6), dtype=np.float32)
    for y in y_ind:
        for x in x_ind:
            x1 = x
            x2 = min(x+blockWidth, width)
            y1 = y
            y2 = min(y+blockHeight, height)
            img = im[y1:y2, x1:x2]
            dets = _detect(sess, net, img)
            dets[:,0] += x1
            dets[:,1] += y1
            dets[:,2] += x1
            dets[:,3] += y1
            all_dets = np.vstack((all_dets, dets)) 
    
    NMS_THRESH = 0.1
    print((len(all_dets)))

    keep = nms(all_dets, NMS_THRESH)
    all_dets = all_dets[keep, :]

    print((len(all_dets)))
    return all_dets        

def detect(net, imgPath, isblock=False):
    global sess
    im = cv2.imread(imgPath)
    timer = Timer()
    timer.tic()
    
    # detect blocks in image
    all_dets = _block_im_detect(sess, net, im)

    timer.toc()
    print((('Detection took {:.3f}s for '
           '{:d} object proposals, {}').format(timer.total_time, all_dets.shape[0], imgPath)))

    for i in range(all_dets.shape[0]):
        bbox = all_dets[i, :4]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 3)

    return im
