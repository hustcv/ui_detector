import numpy as np
import cv2
from math import *
from operator import itemgetter
import pytesseract
from PIL import Image
import sys

def dumpRotateImage(img,degree,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))

    imgOut=imgRotation[int(pt1[1]):int(pt3[1]),int(pt1[0]):int(pt3[0])]
    height,width=imgOut.shape[:2]

    return imgOut

def getTextRoughRoi(rgb):
    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create(_delta=7,_min_area=30,_max_area=200,_max_variation = 0.25)
    msers,bboxes = mser.detectRegions(gray)

    hh,ww = gray.shape[:2]
    ccmap = np.zeros(gray.shape,np.uint8)
    wexpr = 0.3
    cnt = 0
    for indx in xrange(bboxes.shape[0]):
        x,y,w,h = bboxes[indx]
        ratio = float(h)/w
        if ratio > 1.8 and ratio < 4:
            #cv2.rectangle(rgb, (x,y),(x+w, y+h), (255, 0, 0), 2)
            cnt += 1
            xl = int(max(0,x-wexpr*w))
            xr = int(min(ww-1,x+w*(1+wexpr)))
            cv2.rectangle(ccmap, (xl,y),(xr, y+h), (255, 0, 0), -1)

    #cv2.imshow('',rgb)
    #cv2.waitKey()

    print('valid msers:{}'.format(cnt))

    im,contours,hierarchy = cv2.findContours(ccmap,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    cands_ = []
    hlist = []
    for ccs in contours:
        x,y,w,h = cv2.boundingRect(ccs)
        rect = cv2.minAreaRect(ccs)
        #print cv2.contourArea(ccs),float(w)/h
        if cv2.contourArea(ccs) > 150 and float(w)/h > 0.99:
            cands.append((x,y,w,h))
            cands_.append(rect)
            hlist.append(h)

    if len(hlist) == 0:
        return None

    print('candidate areas:{}'.format(len(hlist)))

    x,y,w,h = cands[hlist.index(max(hlist))]
    rect = cands_[hlist.index(max(hlist))]
    #cv2.rectangle(rgb,(x,y),(x+w,y+h),(255,0,0),2)

    print('remains:1')

    hexpr = 0
    y1 = max(0,y-hexpr)
    y2 = min(hh,y+h+hexpr)
    rgb1 = rgb[y1:y2,x:x+w]
    gray1 = gray[y1:y2,x:x+w]

    rect_ = rect
    angle = rect[2]
    rw = rect[1][0]
    rh = rect[1][1]
    if rect[1][0] < rect[1][1]:
        angle = rect[2] + 90 if rect[2] < 0 else rect[2] - 90
        rw = rect[1][1]
        rh = rect[1][0]
        rect_ = (rect[0],[rw,rh],angle)

    if angle == -90.0:
        angle = 0.0
    rect_ = (rect_[0],rect_[1],angle)

    box = cv2.boxPoints(rect_)
    box = np.int0(box)
    #cv2.drawContours(rgb,[box],0,(0,0,255),2)

    print('<Horizontal tilt correction>')
    print('angle:{}'.format(angle))

    roi = dumpRotateImage(rgb,angle,(box[1,0],box[1,1]),(box[2,0],box[2,1]),(box[3,0],box[3,1]),(box[0,0],box[0,1]))

    #roi = []
    #if angle >= 0.0:
    #    roi = dumpRotateImage(rgb,angle,(box[1,0],box[1,1]),(box[2,0],box[2,1]),(box[3,0],box[3,1]),(box[0,0],box[0,1]))
    #else:
    #    roi = dumpRotateImage(rgb,angle,(box[0,0],box[0,1]),(box[1,0],box[1,1]),(box[2,0],box[2,1]),(box[3,0],box[3,1]))

    return roi

def getTextFineRoi(rgb):
    #rgb = cv2.imread('boxes2/000011_0.jpg')
    #src = cv2.imread('tproi/000002_0.jpg')

    if rgb.shape[0] < 32:
        ratio = 32.0/rgb.shape[0]
        rgb = cv2.resize(rgb,None,fx=ratio,fy=ratio)

    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

    bina = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,0)
    im,contours,hierarchy = cv2.findContours(bina,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print len(contours)
    isBright = True
    ncc = 0
    for ccs in contours:
        _,_,w,h = cv2.boundingRect(ccs)
        #print float(h)/w
        if cv2.contourArea(ccs) > rgb.shape[0] and float(h)/w > 1.05 and float(h)/w < 5:
            ncc += 1
    #print ncc

    if ncc < 3:
        isBright = False
        bina = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,0)
        im,contours,hierarchy = cv2.findContours(bina,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #print len(contours)
        ncc = 0
        for ccs in contours:
            _,_,w,h = cv2.boundingRect(ccs)
            if cv2.contourArea(ccs) > rgb.shape[0] and float(h)/w > 1.05 and float(h)/w < 5:
                ncc += 1
        #print ncc

    #cv2.imshow('',bina)
    #cv2.waitKey()

    if isBright:
        print('color:bright')
    else:
        print('color:dark')

    print('rois:{}'.format(ncc))

    roi = None
    if ncc >= 3:
        cori = []
        for ccs in contours:
            x,y,w,h = cv2.boundingRect(ccs)
            rect = cv2.minAreaRect(ccs)
            #print cv2.contourArea(ccs),float(h)/w
            if cv2.contourArea(ccs) > rgb.shape[0] and float(h)/w > 1.05 and float(h)/w < 5:
                angle = rect[2]
                if rect[1][0] > rect[1][1]:
                    angle = rect[2] + 90 if rect[2] < 0 else rect[2] - 90
                cori.append(angle)
        
        #print cori
        angle = np.median(cori)
        #angle = angle - 90 if angle > 0 else angle + 90
        if isBright is False:
            gray = 255 - gray

        height,width=gray.shape[:2]
        widthNew = int(height * fabs(tan(radians(angle))) + width)
        matRotation = np.float32([[1,tan(radians(angle)),0],[0,1,0]])
        if angle < 0:
            matRotation = np.float32([[1,tan(radians(angle)),-height*tan(radians(angle))],[0,1,0]])
        gray_ = cv2.warpAffine(gray, matRotation, (widthNew, height))#, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        rgb_ = cv2.warpAffine(rgb, matRotation, (widthNew, height))

        wext = max(1,int(height*fabs(tan(radians(angle)))))
        gray_ = gray_[:, wext:-wext]
        rgb_ = rgb_[:, wext:-wext]

        print('<Vertical tilt correction>')
        print('angle:{}\nshape:{}'.format(angle,rgb_.shape))

        bina = cv2.adaptiveThreshold(gray_,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,0)
        im,contours,hierarchy = cv2.findContours(bina,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        ncc = 0
        #print len(contours)
        rlist = []
        for ccs in contours:
            x,y,w,h = cv2.boundingRect(ccs)
            rect = cv2.minAreaRect(ccs)
            #print cv2.contourArea(ccs),h,float(h)/w
            if cv2.contourArea(ccs) > rgb.shape[0] and float(h)/w > 1.05 and float(h)/w < 5:
                rlist.append((x+w/2,y+h/2,w,h,float(h)/w))
                #cv2.rectangle(rgb_,(x,y),(x+w,y+h),(0,255,0),1)
                ncc += 1
        #print ncc

        #cv2.imshow('',bina)
        #cv2.waitKey()

        print('rois:{}'.format(ncc))

        if ncc >= 3:
            rlist_ = sorted(rlist,key=itemgetter(0))
            #print rlist_

            rlist2 = []
            rlist2_ = [0]
            rlen = []
            for ind in xrange(len(rlist_)-1):
                if rlist_[ind][0]+rlist_[ind][2]*0.8 > rlist_[ind+1][0]-rlist_[ind+1][2]*0.8:
                    rlist2_.append(ind+1)
                else:
                    rlist2.append(rlist2_)
                    rlen.append(len(rlist2_))
                    rlist2_ = [ind+1]

            if rlist2_ is not None:
                rlist2.append(rlist2_)
                rlen.append(len(rlist2_))

            clist = []
            if len(rlen) > 1:
                clist = rlist2[rlen.index(max(rlen))]
            else:
                clist = rlist2[0]

            rpts = []
            #cpts = []
            if len(clist) >= 3:
                rlist3 = [rlist_[rg][4] for rg in clist]
                hlist3 = [rlist_[rg][3] for rg in clist]

                rmean = np.median(rlist3)
                hmean = np.median(hlist3)
                #rstd = np.std(rlist3)
                #rlow,rhigh = rmean-1.95*rstd,rmean+1.95*rstd
                rlow,rhigh = rmean-1,rmean+1
                hlow,hhigh = hmean*0.8,hmean*1.2

                noi = 0
                for rg in clist:
                    if rlist_[rg][4] > rlow and rlist_[rg][4] < rhigh and rlist_[rg][3] > hlow and rlist_[rg][3] < hhigh:
                        noi += 1
                        if noi < 6:
                            rpts += [rlist_[rg][0]-rlist_[rg][2]/2,rlist_[rg][1]-rlist_[rg][3]/2,rlist_[rg][0]+rlist_[rg][2]/2,rlist_[rg][01]+rlist_[rg][3]/2]
                            #cpts += [rlist_[rg][0],rlist_[rg][1]]
            
                #print type(rpts),rpts
                
                if noi > 4:
                    if rpts[1] < 2 and rpts[5] > 2 and rpts[9] > 2 and rpts[13] > 2:
                        rpts = rpts[4:]
                        #cpts = cpts[2:] 
                    else:
                        rpts = rpts[:-4]
                        #cpts = cpts[:-2]     
                
            else:
                for rg in clist:
                    rpts += [rlist_[rg][0]-rlist_[rg][2]/2,rlist_[rg][1]-rlist_[rg][3]/2,rlist_[rg][0]+rlist_[rg][2]/2,rlist_[rg][01]+rlist_[rg][3]/2]
                    #cpts += [rlist_[rg][0],rlist_[rg][1]]

            rpts_ = np.array(rpts).reshape(-1,2)
            #print rpts_
            x,y,w,h = cv2.boundingRect(rpts_)
            roi = gray_[y:y+h,x:x+w]

            print('Number area generation>')
            print('rect:({} {} {} {})'.format(x,y,w,h))

            #cpts_ = np.array(cpts).reshape(-1,2)
            #print cpts_
            #[vx,vy,x,y] = cv2.fitLine(cpts_,cv2.DIST_L2,0,0.01,0.01)
            #angle = degrees(np.arctan(vy/vx))

            #roi = roi_

    #if roi is not None:
    #    cv2.imshow('',roi)
    #    #cv2.imwrite('roi.jpg',roi)
    #    cv2.waitKey()
    
    return roi

def roi2str_tess(rgb):
    im = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

    ratio = 90.0/im.shape[0]
    im_ = cv2.resize(im,None,fx=ratio,fy=ratio)

    bina = cv2.adaptiveThreshold(im_,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,0)
    bina_,contours,hierarchy = cv2.findContours(bina,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    ncc = 0
    for ccs in contours:
            _,_,w,h = cv2.boundingRect(ccs)
            if cv2.contourArea(ccs) > 10*im.shape[0] and float(h)/w > 1.05:
                ncc += 1
                #cv2.drawContours(bina_,[ccs],0,0,-1)

    if ncc < 4:
        bina = cv2.adaptiveThreshold(im_,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,0)
        bina_,contours,hierarchy = cv2.findContours(bina,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        ncc = 0
        for ccs in contours:
                _,_,w,h = cv2.boundingRect(ccs)
                if cv2.contourArea(ccs) > 10*im.shape[0] and float(h)/w > 1.05:
                    ncc += 1

    bina_,contours,hierarchy = cv2.findContours(bina,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cori = []
    for ccs in contours:
        _,_,w,h = cv2.boundingRect(ccs)
        if cv2.contourArea(ccs) < 10*im.shape[0] or float(h)/w < 1.05:
            cv2.drawContours(bina_,[ccs],0,0,-1)
        else:
            rect = cv2.minAreaRect(ccs)
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle = rect[2] + 90 if rect[2] < 0 else rect[2] - 90
            cori.append(angle)


    #print cori
    angle = np.median(cori)
    #print angle
    angle = angle - 90 if angle > 0 else angle + 90
    #print angle

    if fabs(angle-90) > 6 and fabs(angle) > 6:
        height,width=im_.shape[:2]
        widthNew = int(height * fabs(tan(radians(angle))) + width)
        matRotation = np.float32([[1,tan(radians(angle)),0],[0,1,0]])
        if angle < 0:
            matRotation = np.float32([[1,tan(radians(angle)),-height*tan(radians(angle))],[0,1,0]])
        bina_ = cv2.warpAffine(bina, matRotation, (widthNew, height))

        ret,bina_ = cv2.threshold(bina_,127,255,cv2.THRESH_BINARY)

    '''
    if fabs(angle-90) > 6 and fabs(angle) > 6:
        height,width=im.shape[:2]
        widthNew = int(height * fabs(tan(radians(angle))) + width)
        matRotation = np.float32([[1,tan(radians(angle)),0],[0,1,0]])
        if angle < 0:
            matRotation = np.float32([[1,tan(radians(angle)),-height*tan(radians(angle))],[0,1,0]])
        im = cv2.warpAffine(im, matRotation, (widthNew, height))
        cv2.imwrite('img/rst/000995_0-2.jpg',im)
    '''

    image = Image.fromarray(bina_)
    #image.show()
    code = pytesseract.image_to_string(image,config='-psm 7')

    return code

def roi2str_tess2(gray):
    ratio = 90.0/gray.shape[0]
    gray_ = cv2.resize(gray,None,fx=ratio,fy=ratio)

    bina = cv2.adaptiveThreshold(gray_,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,0)
    bina_,contours,hierarchy = cv2.findContours(bina,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cori = []
    for ccs in contours:
        _,_,w,h = cv2.boundingRect(ccs)
        if cv2.contourArea(ccs) < 10*gray.shape[0] or float(h)/w < 1.05:
            cv2.drawContours(bina_,[ccs],0,0,-1)

    image = Image.fromarray(bina_)
    #image.show()
    code = pytesseract.image_to_string(image,config='-psm 7')

    return code

'''
if __name__ == "__main__":
    fimg = '/workspace/data/JPEGImages_text/'+sys.argv[1]+'.jpg'
    rgb = cv2.imread(fimg)
'''

def getTPNumber(rgb):
    if rgb.shape[0] > 200:
        rgb = cv2.resize(rgb,None,fx=0.6,fy=0.6)

    print('<Extract rough text area>')
    roi_r = getTextRoughRoi(rgb)
    
    #cv2.imshow('',roi_r)
    #cv2.waitKey()

    if roi_r is None:
        return None

    tpnum = None
    if roi_r is not None or roi_r.shape[0] > 0:
        print('shape:{}'.format(roi_r.shape))
        print('<Extract fine text/number area>')
        roi = getTextFineRoi(roi_r)

        if roi is not None:
            #cv2.imwrite('roi.jpg',roi)
            #cv2.imshow('',roi)
            #cv2.waitKey()

            print('<Recognize numbers>')
            tpnum = roi2str_tess2(roi)
            print tpnum

        else:
            print('detections:0')

    return tpnum