#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is only needed for Python v2 but is harmless for Python v3.
import sip
sip.setapi('QString', 2)

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QTimer, Qt

import sys, os
import time

import dianwang_backend as demo
import numpy as np
import cv2
import Queue

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        """ Constructor initializes a default value for the brightness, creates
            the main menu entries, and constructs a central widget that contains
            enough space for images to be displayed.
        """
        super(MainWindow, self).__init__()

        self.net = demo.Detector()

        self.img_ret = np.array([])
        self.scaledImage = QtGui.QImage()
        self.path = ''
        self.fileName = []
        self.directoryFile = ''
        self.step = 0
        self.count=0
        self.queue_bbox = Queue.Queue()

        # ui setting
        self.width = 800
        self.height = 600

        self.resize(1200, 800)
        self.setWindowTitle("Dianwang Detection")
            
        # image list
        self.listView = QtGui.QListView()
        self.listView.setMaximumWidth(200)
        self.listViewModel = QtGui.QStandardItemModel(self.listView)

        legendFile = "legend.jpg"
        self.legend = QtGui.QLabel()
        legendImage = QtGui.QImage()
        if legendImage.load(legendFile):
            self.legend.setPixmap(QtGui.QPixmap.fromImage(legendImage))
        leftLayout = QtGui.QVBoxLayout()
        leftLayout.addWidget(self.listView)
        leftLayout.addWidget(self.legend)

        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setFrameShadow(QtGui.QFrame.Sunken)
        self.imageLabel.setFrameShape(QtGui.QFrame.StyledPanel)
        self.imageLabel.setMinimumSize(QtCore.QSize(256, 256))

        self.resultBox = QtGui.QGridLayout()
        self.towerLabel        = QtGui.QLabel(u"塔") 
        self.insulatorLabel    = QtGui.QLabel(u"绝缘子") 
        self.hammerLabel       = QtGui.QLabel(u"防震锤") 
        self.nestLabel         = QtGui.QLabel(u"鸟巢") 
        self.textLabel         = QtGui.QLabel(u"塔号牌") 
        self.textLabel_rec     = QtGui.QLabel(u"塔号识别") 
        self.badInsulatorLabel = QtGui.QLabel(u"受损绝缘子")
        self.badHammerLabel    = QtGui.QLabel(u"受损防震锤")
    
        self.towerNumLabel        = QtGui.QLabel('0')
        self.insulatorNumLabel    = QtGui.QLabel('0')
        self.hammerNumLabel       = QtGui.QLabel('0')   
        self.nestNumLabel         = QtGui.QLabel('0')
        self.textNumLabel         = QtGui.QLabel('0')
        self.textStrLabel         = QtGui.QLabel('null')
        self.badInsulatorNumLabel = QtGui.QLabel('0')
        self.badHammerNumLabel    = QtGui.QLabel('0')

        self.resultBox.addWidget(self.towerLabel,          0,0,1,1) 
        self.resultBox.addWidget(self.towerNumLabel,       0,1,1,1)   
        self.resultBox.addWidget(self.nestLabel,           0,2,1,1)
        self.resultBox.addWidget(self.nestNumLabel,        0,3,1,1)  
        self.resultBox.addWidget(self.textLabel,           0,4,1,1)
        self.resultBox.addWidget(self.textNumLabel,        0,5,1,1)      
        self.resultBox.addWidget(self.textLabel_rec,       0,6,1,1)
        self.resultBox.addWidget(self.textStrLabel,        0,7,1,1)  

        self.resultBox.addWidget(self.insulatorLabel,      1,0,1,1)    
        self.resultBox.addWidget(self.insulatorNumLabel,   1,1,1,1)   
        self.resultBox.addWidget(self.badInsulatorLabel,   1,2,1,1)
        self.resultBox.addWidget(self.badInsulatorNumLabel,1,3,1,1) 

        self.resultBox.addWidget(self.hammerLabel,         2,0,1,1)
        self.resultBox.addWidget(self.hammerNumLabel,      2,1,1,1)            
        self.resultBox.addWidget(self.badHammerLabel,      2,2,1,1) 
        self.resultBox.addWidget(self.badHammerNumLabel,   2,3,1,1) 

        rightLayout = QtGui.QVBoxLayout()
        rightLayout.addWidget(self.imageLabel, 8)
        rightLayout.addLayout(self.resultBox, 1)

        self.openButton = QtGui.QPushButton("Open")
        self.openButton.setMinimumSize(64, 32)
        self.openButton.setMaximumWidth(200)
        
        self.detectButton = QtGui.QPushButton("Detect")
        self.detectButton.setMinimumSize(64,32)

        #self.saveButton = QtGui.QPushButton("Save")
        #self.saveButton.setMinimumSize(32, 32)

        self.quitButton = QtGui.QPushButton("Quit")
        self.quitButton.setMinimumSize(64, 32)

        self.pbar = QtGui.QProgressBar(self)
        self.pbar.setMinimumSize(64,32)

        self.pbar_label = QtGui.QLabel('ProgressBar')
        #self.pbar.resize(20,20)

        frame = QtGui.QFrame(self)
        grid = QtGui.QGridLayout(frame)
        grid.setSpacing(10)

        grid.addLayout(leftLayout,       0, 0, 1,0)
        grid.addLayout(rightLayout,      0, 1, 1,2)
        grid.addWidget(self.openButton,  1, 0)
        grid.addWidget(self.detectButton,1, 1)
        grid.addWidget(self.quitButton,  1, 2) 
        grid.addWidget(self.pbar_label,  2, 0)
        grid.addWidget(self.pbar,        2, 1, 1, 2)
        self.setCentralWidget(frame)

        #########################
        # connect signal and slot
        self.openButton.clicked.connect(self.chooseFile)

        self.detectButton.clicked.connect(self.detect_one)
        #self.detectButton.clicked.connect(self.begin_detect)

        #self.saveButton.clicked.connect(self.saveImage)
        self.quitButton.clicked.connect(self.close)

        self.timer = QTimer(self)
        self.sum_delay = 2000
        self.timer.setInterval(self.sum_delay)
        self.timer.timeout.connect(self.detect_one)
        self.timer.setSingleShot(True)

        self.show_bbox_timer = QTimer(self)
        self.delay = 1000
        self.show_bbox_timer.setInterval(self.delay)
        self.show_bbox_timer.setSingleShot(True)
        self.show_bbox_timer.timeout.connect(self.show_one_bbox)

    def chooseFile(self):
        """ Provides a dialog window to allow the user to specify an image file.
            If a file is selected, the appropriate function is called to process
            and display it.
        """

        #imageFile = QtGui.QFileDialog.getOpenFileName(self,
        #      "Choose an image file to open", self.path, "Images (*.*)")
        
        directoryFile = QtGui.QFileDialog.getExistingDirectory()
        self.directoryFile = directoryFile
        
        FileName = []
        FileName = os.listdir(directoryFile)
        FileName = sorted([x for x in FileName if x[-3:].lower() in ['jpg', 'png']])
        self.fileName = FileName

        # set imgname in listView
        self.listViewModel.clear()
        for name in FileName:
            item = QtGui.QStandardItem(name)
            self.listViewModel.appendRow(item)
        self.listView.setModel(self.listViewModel)

        self.step = 0
        self.pbar.setMinimum(0)    
        self.pbar.setMaximum(len(self.fileName))
        self.phase = "show"

    def openImageFile(self, imageFile):
        originalImage = QtGui.QImage()
        if originalImage.load(imageFile):
            self.setWindowTitle(imageFile)
            self.scaledImage = originalImage.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.scaledImage))
        else:
            QtGui.QMessageBox.warning(self, "Cannot open file",
                    "The selected file could not be opened.",
                    QtGui.QMessageBox.Cancel, QtGui.QMessageBox.NoButton,
                    QtGui.QMessageBox.NoButton)

    def begin_detect(self):
        self.timer.start()

    def detect_one(self):
        if self.step == len(self.fileName):
            print("please choose new images")
            self.timer.stop()
            return

        self.openImageFile(os.path.join(self.directoryFile, self.fileName[self.step]))

        # detect 
        self.path = os.path.join(self.directoryFile, self.fileName[self.step])
        self.img_ret, all_dets, data_ret = self.net.detect(self.path)

        # put all_dets into queue
        for i in range(all_dets.shape[0]):
            bbox = all_dets[i]
            self.queue_bbox.put(bbox)

        # show result
        self.towerNumLabel.setText(str(data_ret['tower']))
        self.insulatorNumLabel.setText(str(data_ret['insulator']))
        self.hammerNumLabel.setText(str(data_ret['hammer']))
        self.nestNumLabel.setText(str(data_ret['nest']))
        self.textNumLabel.setText(str(data_ret['text']))
        self.badInsulatorNumLabel.setText(str(data_ret['bad_insulator']))
        self.badHammerNumLabel.setText(str(data_ret['bad_hammer']))
        self.textStrLabel.setText(str(data_ret['textStr']))

        ## display img, box one by one
        self.show_bbox_timer.start()

        self.step += 1
        self.pbar.setValue(self.step)

    def show_one_bbox(self):
        if self.queue_bbox.empty():
            return;

        bbox = self.queue_bbox.get()
        class_name = demo.cls_map[str(int(bbox[-1]))]
       # cv2.rectangle(self.img_ret, (bbox[0], bbox[1]), (bbox[2], bbox[3]), demo.color[class_name], 16)
       # img_roi=cv.fromarray(self.img_ret)
       # img_roi=cv2.imdecode(self.img_ret,cv2.IMREAD_COLOR)
      # img_roi=self.img_ret[int(bbox[0]):int(bbox[2]),int(bbox[1]):int(bbox[3]),:]
       # path1="insulator_roi"
        print bbox[0],bbox[1],bbox[2],bbox[3]
        height, width, channel = self.img_ret.shape
        img_roi=self.img_ret[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
        path1="/home/cv/insulator_roi"
        filetype=".jpg"
        newdir=os.path.join(path1,str(self.count).zfill(3)+filetype)
        print(newdir, img_roi.shape)
        cv2.imwrite(newdir,img_roi)
        self.count+=1
        cv2.rectangle(self.img_ret,(bbox[0],bbox[1]),(bbox[2],bbox[3]),demo.color[class_name],16)
       # print height,width
        bytesPerLine = channel * width
        resultImg = QtGui.QImage(self.img_ret.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.scaledImage = resultImg.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.scaledImage))
        
        if not self.queue_bbox.empty():
            self.show_bbox_timer.start()
        else:
            self.timer.start()


    def saveImage(self,imagename):
        """ Provides a dialog window to allow the user to save the image file.
        """
        #imageFile = QtGui.QFileDialog.getSaveFileName(self,
        #        "Choose a filename to save the image", "", "Images (*.png)")
        #print(imageFile)
        #info = QtCore.QFileInfo(imageFile)
        info = imagename
        #if info.baseName() != '':
        if info !='':
           # print(info)
           # newImageFile = QtCore.QFileInfo(info.absoluteDir(),
            #        info+'detected' + '.png').absoluteFilePath()
            newImageFile = '/home/fangfang/py-faster-rcnn/insulator/'+info
            print(newImageFile)
            if not self.imageLabel.pixmap().save(newImageFile, 'PNG'):
                QtGui.QMessageBox.warning(self, "Cannot save file",
                        "The file could not be saved.",
                        QtGui.QMessageBox.Cancel, QtGui.QMessageBox.NoButton,
                        QtGui.QMessageBox.NoButton)
        else:
            QtGui.QMessageBox.warning(self, "Cannot save file",
                    "Please enter a valid filename.", QtGui.QMessageBox.Cancel,
                    QtGui.QMessageBox.NoButton, QtGui.QMessageBox.NoButton)


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
