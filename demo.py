import numpy as np
import cv2
import sys

import detect_mser as textreader

if __name__ == "__main__":
    #fimg = '/workspace/data/JPEGImages_text/' + sys.argv[1] + '.jpg'
    fimg = '/home/cv/insulator_roi/000000.jpg'
    rgb = cv2.imread(fimg)

    tpnum = textreader.getTPNumber(rgb)
    if tpnum == None:
        print 'TP Number: '
    else:
        print 'TP Number: ' + tpnum

