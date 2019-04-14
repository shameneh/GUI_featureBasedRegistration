import sys
from PySide2.QtWidgets import *#QApplication, QPushButton, QMainWindow,QDesktopWidget()
from PySide2.QtCore import Slot ,Signal,Qt, QPoint
from PySide2.QtGui import QImage, QPixmap,QPalette

#Registration part
import os
#import specto_backend as sb
from deskew import deskew
import cv2
import numpy as np
import ImageRegistration as imReg
#import matplotlib.pyplot as plt

#%matplotlib inline


# class RightClickMenuButton(QPushButton):
#
#     def __init__(self, name, parent = None):
#         super(RightClickMenuButton, self).__init__(name)
#
#         self.setContextMenuPolicy(Qt.ActionsContextMenu)
#         self.addMenuActions();
#
#     def addMenuActions(self):
#         delete = QAction(self)
#         delete.setText("remove")
#         delete.triggered.connect(self.removeButton)
#         self.addAction(delete)
#
#     def removeButton(self):
#         self.deleteLater()

# dw=QtGui.QDesktopWidget()
# dw.screenGeometry()
# dw.availableGeometry() # this is a sub rect of screenGeometry because it e.g. ignores the space occupied by the task bar on Windows
class MyGui(QWidget):
    def __init__(self):
        super(MyGui, self).__init__()
        dw = QDesktopWidget()
        self.setFixedSize(dw.width()*1,dw.height()*0.9)
        self.setWindowTitle("Neuro Soph - Image Registration")
        self.height = 700
        self.width  = 420

        btn_selectMV = QPushButton("Select Moving Image")

        # btn_selectMV.setText("Select Moving Image")
        btn_selectMV.clicked.connect(self.selectMovingImage)

        # elf.btn_selectT = QToolButton(self)
        btn_selectT = QPushButton("Select Fixed Image")
        # btn_selectT.setText("Select Fixed Image")
        btn_selectT.clicked.connect(self.selectFixedImage)
        self.filenameFixed =''#''/Users/aminoo/Documents/Work/ImageRegistration/templates/Template_P01.png'
        self.filenameMoving=''#/Users/aminoo/Documents/Work/ImageRegistration/Data/Small_053/Page1/P01_Images/P01_00013_E__IM001.png'
        ###images
        self.coderun =False
        self.x = -1
        self.y = -1

        TextF = QLabel('Fixed Image')
        TextM = QLabel('Moving Image')
        TextR = QLabel('Result Image')
        self.fixedImageLB = QLabel()
        self.fixedImageLB.setScaledContents(False);
        self.fixedImageLB.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        #self.fixedImageLB.ad

        self.movingImageLB= QLabel()
        self.movingImageLB.setScaledContents(False);

        self.resultLB = QLabel()
        self.resultLB.setScaledContents(False);
        self.resultQI = QImage()
        self.fixedImageQI = QImage()
        self.movingImageQI = QImage()
        self.scrollf = QScrollArea()
        self.scrollf.setWidget(self.fixedImageLB)
        self.scrollf.setVisible(True)
        self.scrollf.setWidgetResizable(True)
        self.scrollf.setFixedHeight(self.height)
        self.scrollf.setFixedWidth(self.width)

        self.scrollr = QScrollArea()
        self.scrollr.setWidget(self.resultLB)
        self.scrollr.setVisible(True)
        self.scrollr.setWidgetResizable(True)
        self.scrollr.setFixedHeight(self.height)
        self.scrollr.setFixedWidth(self.width)

        self.scrollm = QScrollArea()
        self.scrollm.setWidget(self.movingImageLB)
        self.scrollm.setVisible(True)
        self.scrollm.setWidgetResizable(True)
        self.scrollm.setFixedHeight(self.height)
        self.scrollm.setFixedWidth(self.width)

        self.PAGE_DPI=150


        #Right Click
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        self.addMenuActions()
       # self.init_page()
        ####
        self.textscore_AKA= QLabel()
        self.textscore_SIF = QLabel()
        button1 = QPushButton("AKAZE")
        button3 = QPushButton("RegisterAgain")
        button2 = QPushButton("SIFT")
        # button4 = QPushButton("Four")
        # button5 = QPushButton("Five")

        button1.clicked.connect(self.AKAZERegistration)
        button3.clicked.connect(self.try_registeragain)
        button2.clicked.connect(self.SIFTRegistration)
        mainhbox = QHBoxLayout()

        lFvbox = QVBoxLayout()
        lFvbox.addWidget(TextF)
        lFvbox.addWidget(self.scrollf)

        lMvbox = QVBoxLayout()
        lMvbox.addWidget(TextM)
        lMvbox.addWidget(self.scrollm)


        lRvbox = QVBoxLayout()
        lRvbox.addWidget(TextR)
        lRvbox.addWidget(self.scrollr)

        imagehbox = QHBoxLayout()
        imagehbox.addLayout(lFvbox)
        imagehbox.addLayout(lMvbox)
        imagehbox.addLayout(lRvbox)



        layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addWidget(self.textscore_AKA)
        layout.addWidget(self.textscore_SIF)
        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)

        # layout.addWidget(button4)
        # layout.addWidget(button5)


       # mainhbox.addWidget(RightClickMenuButton("Test Btn"))
        mainhbox.addLayout(layout)
        mainhbox.addLayout(imagehbox)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(btn_selectT)
        hbox.addWidget(btn_selectMV)



        mainvbox = QVBoxLayout()
        mainvbox.addStretch(1)
        mainvbox.addLayout(mainhbox)
        mainvbox.addLayout(hbox)

        self.setLayout(mainvbox)


        self.show()



    def addMenuActions(self):
        menu = QMenu(self)
        delete = QAction(self)
        delete.setText("Remove")
        delete.triggered.connect(self.removeFeature)

        insert = QAction(self)
        insert.setText("Insert")
        insert.triggered.connect(self.insertFeature)

        cancel = QAction(self)
        cancel.setText("Cancel")
        cancel.triggered.connect(self.cancel)
        self.addAction(delete)
        self.addAction(insert)
        self.addAction(cancel)



    def removeFeature(self):

        if(self.x!=-1 and self.y!=-1):
            print(self.x, self.y)

            p1 = QPoint()
            p1.setX(self.x)  # point.pt[0]
            p1.setY(self.y) # point.pt[1]

            self.selectedF = False

            if self.x <= self.scrollf.x()+self.width and self.x >= self.scrollf.x() and self.y <= self.scrollf.y()+self.height and self.y >= self.scrollf.y():

                rv = self.mapToGlobal(p1)
                self.newp = self.fixedImageLB.mapFromGlobal(rv)
                print("selected ", self.newp.x(), self.newp.y())
                print('firstkeyf ', len(self.keypoints[0]))
                cv2.circle(self.imgff, (int(self.newp.x()), int(self.newp.y())), 5, (0, 0, 0), -1)

                height, width, channel = self.imgff.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.imgff, width, height, bytesPerLine, QImage.Format_RGB888)
                self.fixedImageQI = qImg
                pixMapf = QPixmap.fromImage(self.fixedImageQI)
                self.fixedImageLB.setPixmap(pixMapf)
                newk=[]
                for point in self.keypoints[0]:
                    if (int(point.pt[0]) <= int(self.newp.x())+2 and int(point.pt[0]) >=int(self.newp.x())-2):
                        if(int(point.pt[1]) <= int(self.newp.y())+2 and int(point.pt[1]) >= int(self.newp.y())-2):
                            print('yes')
                            self.selectedF=True
                        else:
                            newk.append(point)
                    else:
                        newk.append(point)
                if self.selectedF:
                    self.keypoints[0]=newk
            elif self.x <=self.scrollm.x()+self.width and self.x >= self.scrollm.x()and self.y <= self.scrollm.y()+self.height and self.y >= self.scrollm.y():
                self.selectedM=False
                rvm = self.mapToGlobal(p1)
                self.newpm = self.movingImageLB.mapFromGlobal(rvm)
                print("selected ", self.newpm.x(), self.newpm.y())
                print('firstkeym ', len(self.keypointsM))
                cv2.circle(self.imgmm, (int(self.newpm.x()), int(self.newpm.y())), 5, (0, 0, 0), -1)

                height, width ,channel= self.imgmm.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.imgmm, width, height, bytesPerLine, QImage.Format_RGB888)
                self.movingImageQI = qImg
                pixMapm = QPixmap.fromImage(self.movingImageQI)
                self.movingImageLB.setPixmap(pixMapm)
                newmk = []
                for pointm in self.keypointsM:
                    if (int(pointm.pt[0]) <= int(self.newpm.x()) + 2 and int(pointm.pt[0]) >= int(self.newpm.x()) - 2):
                        if (int(pointm.pt[1]) <= int(self.newpm.y()) + 2 and int(pointm.pt[1]) >= int(self.newpm.y()) - 2):
                            print('yes')  # remove this point
                            self.selectedM=True
                        else:
                            newmk.append(pointm)
                    else:
                        newmk.append(pointm)
                if self.selectedM:
                    self.keypointsM= newmk
                print('keym', len(self.keypointsM),len(self.descriptorsM))
    def insertFeature(self):
        if (self.x != -1 and self.y != -1):
            print(self.x, self.y)
            p1 = QPoint()
            p1.setX(self.x)
            p1.setY(self.y)

            if self.x <= self.scrollf.x()+self.width and self.x >= self.scrollf.x() and self.y <= self.scrollf.y()+self.height and self.y >= self.scrollf.y():
                rv = self.mapToGlobal(p1)
                self.newp = self.fixedImageLB.mapFromGlobal(rv)
                print("selected ", self.newp.x(), self.newp.y())
                print('firstkeyf ', len(self.keypoints[0]))
                cv2.circle(self.imgff, (int(self.newp.x()), int(self.newp.y())), 5, (0, 0, 0), -1)

                height, width, channel = self.imgff.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.imgff, width, height, bytesPerLine, QImage.Format_RGB888)
                self.fixedImageQI = qImg
                pixMapf = QPixmap.fromImage(self.fixedImageQI)
                self.fixedImageLB.setPixmap(pixMapf)
                newpf=QPoint()
                newpf.setX(self.newp.x())
                newpf.setY(self.newp.y())
                # newifk=[]
                # for pointm in self.keypoints[0]:
                #     newifk.append(pointm)
                # newifk.append(newpf)
                self.keypoints[0].append(cv2.KeyPoint(self.newp.x(), self.newp.y(), 5, _class_id=0))

            elif self.x <= self.scrollm.x()+self.width and self.x >= self.scrollm.x()and self.y <=self.scrollm.y() +self.height and self.y >= self.scrollm.y():
                rvm = self.mapToGlobal(p1)
                self.newpm = self.movingImageLB.mapFromGlobal(rvm)
                print("selected ", self.newpm.x(), self.newpm.y())
                print('firstkeym ', len(self.keypointsM))
                cv2.circle(self.imgmm, (int(self.newpm.x()), int(self.newpm.y())), 5, (0, 0, 0), -1)

                height, width, channel = self.imgmm.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.imgmm, width, height, bytesPerLine, QImage.Format_RGB888)
                self.movingImageQI = qImg
                pixMapm = QPixmap.fromImage(self.movingImageQI)
                self.movingImageLB.setPixmap(pixMapm)

                # newpm = QPoint()
                # newpm.setX(self.newpm.x())
                # newpm.setY(self.newpm.y())

                self.keypointsM.append(cv2.KeyPoint(self.newpm.x(), self.newpm.y(), 5, _class_id=0))
                print('s')
    def cancel(self):
        pass

    # def addMenuActions(self):
    #     delete = QAction(self)
    #     delete.setText("remove")
    #     delete.triggered.connect(self.removeButton)
    #     self.addAction(delete)
    #
    # def removeButton(self):
    #     self.deleteLater()
    def mousePressEvent(self, QMouseEvent):
        # print mouse position
        self.x = QMouseEvent.x()
        self.y = QMouseEvent.y()

        # print(self.x,self.y)

    def init_page(self):

        self.fixedImageQI.load(self.filenameFixed)#'/Users/aminoo/Neurosoph/Images/init_image.png')
        self.imf = cv2.imread(self.filenameFixed,flags=0)


        self.movingImageQI.load(self.filenameMoving)#'/Users/aminoo/Neurosoph/Images/init_image.png')



        pixMapF = QPixmap.fromImage(self.fixedImageQI)
        pixMapM = QPixmap.fromImage(self.movingImageQI)


       # self.resultLB.setPixmap(pixMapR)
       # self.resultLB.setMask(pixMapR.mask())
        self.fixedImageLB.setPixmap(pixMapF)
        self.fixedImageLB.setMask(pixMapF.mask())
        self.movingImageLB.setPixmap(pixMapM)
        self.movingImageLB.setMask(pixMapM.mask())





    def AKAZERegistration(self):


        self.method='AKAZE'
        self.registration()

    def SIFTRegistration(self):
        self.method = 'SIFT'
        self.registration()

    def registration(self):
        self.coderun = True  # To right click is activated
        self.load_templates()

        # print("Finished loading template (blank form) images")
        self.register_image()
        # print("Registration score: {0}".format(reg_score))


    def initLabelImages(self):
        pass






    def initButtonImages(self):
        btn_selectMV = QPushButton("Select Fixed Image")
        btn_selectMV.clicked.connect(self.select_image)
        btn_selectT = QPushButton("Select Fixed Image")
        #btn_selectT.setText("Select Fixed Image")
        btn_selectT.clicked.connect(self.select_image)




        hbox = QHBoxLayout()
        #hbox.addStretch(1)
        hbox.addWidget(btn_selectMV)
        hbox.addWidget(btn_selectT)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)





        #file dialog

    def selectFixedImage(self):
        file_dialog = QFileDialog(self)
        # the name filters must be a list
        file_dialog.selectNameFilter("Images (*.png *.jpg)")
        file_dialog.setDirectory('/Users/aminoo/Documents/Work/ImageRegistration/templates')
        # show the dialog
        self.filenameFixed, _ = file_dialog.getOpenFileName(self)
        # self.filenameFixed = '/Users/aminoo/Documents/Work/ImageRegistration/templates/Template_P01.png'
        if (self.filenameFixed != ""):
            # Read an image from file and creates an ARGB32 string representing the image

            im = cv2.imread(self.filenameFixed, flags=0)  # read in image in grayscale
            im = cv2.resize(im, dsize=(637, 825), interpolation=cv2.INTER_LINEAR)
            if len(im.shape) == 2:
                im = cv2.applyColorMap(im, cv2.COLORMAP_BONE)
                # im = cv2.resize(im, (512, 512), cv2.INTER_AREA)
            height, width, channel = im.shape
            bytesPerLine = 3 * width
            qImg = QImage(im, width, height, bytesPerLine, QImage.Format_RGB888)
            self.fixedImageQI = qImg
            self.fixedImageQI.scaledToWidth(width)
            self.fixedImageQI.scaledToHeight(height)
            pixMapf= QPixmap.fromImage(self.fixedImageQI)
            #
            # pixMapM = QPixmap.fromImage(self.movingImageQI)
            self.fixedImageLB.setPixmap(pixMapf)
            # self.movingImageLB.setMask(pixMapM.mask())


    def selectMovingImage(self):
        file_dialog = QFileDialog(self)
        # the name filters must be a list
        file_dialog.selectNameFilter("Images (*.png *.jpg)")
        file_dialog.setDirectory('/Users/aminoo/Documents/Work/ImageRegistration/Data/Small_053/Page1/P01_Images')
        self.filenameMoving, _ = file_dialog.getOpenFileName(self)
        if (self.filenameMoving != ""):
            # Read an image from file and creates an ARGB32 string representing the image
            im = cv2.imread(self.filenameMoving, flags=0)  # read in image in grayscale
            im = cv2.resize(im, dsize=(637, 825), interpolation=cv2.INTER_LINEAR)
            if len(im.shape) == 2:
                im = cv2.applyColorMap(im, cv2.COLORMAP_BONE)
            height, width,channel = im.shape
            bytesPerLine = 3 * width
            qImg = QImage(im, width, height, bytesPerLine, QImage.Format_RGB888)
            self.movingImageQI = qImg
            self.movingImageQI.scaledToWidth(width)
            self.movingImageQI.scaledToHeight(height)
            pixMapm = QPixmap.fromImage(self.movingImageQI)
            self.movingImageLB.setPixmap(pixMapm)


    # function to load templates borrowed from Specto backend
    def load_templates(self):  # os.path.join("..", "templates"), PAGE_DPI=150):

        """Function to load the form template images."""
        # TEMPLATE IMAGES SHOULD BE PROCESSED IN THE SAME MANNER AS FORM PAGE IMAGES!! (e.g. thresholding, inverting)
        self.templates = []  # store images in list
        self.keypoints = []
        self.descriptors = []

        #for file in sorted(os.listdir(TEMPLATES_DIR)):
        im = cv2.imread(self.filenameFixed, flags=0)  # read in image in grayscale
        im = cv2.resize(im, (0, 0), fx=self.PAGE_DPI / 300, fy=self.PAGE_DPI / 300,
                        interpolation=cv2.INTER_LINEAR)  # rescale to working DPI

        im = np.invert(im)  # invert the image
        #img = cv2.resize(im, dsize=(450, 700), interpolation=cv2.INTER_LINEAR)

        [kp1, des1,im1] = imReg.get_keypoints(im, downsample_factor=int(2 * self.PAGE_DPI / 300), method=self.method)

        self.templates.append(im)
        self.keypoints.append(kp1)
        self.descriptors.append(des1)
        self.imf = im1





    # want to be able to visualize aligned images - how about making the template the red channel and the aligned form the green?
    def combine_images(template, form, pixel_thresh=127, debug_out=False):
        # generate an RGB image of the same shape as template

        t_shape = template.shape
        f_shape = form.shape
        minH = min(t_shape[0], f_shape[0])
        minW = min(t_shape[1], f_shape[1])



        result = np.ones((minH, minW, 3), np.uint8) * 255  # init to [255,255,255] everywhere = white image

        result[:minH, :minW, 1] = 255 - 255 * (template[:minH, :minW] < pixel_thresh)  # red channel: template
        result[:minH, :minW, 0] = 255 - 255 * (form[:minH, :minW] < pixel_thresh)  # blue channel: form
        result[:minH, :minW, 2] = 255 - 255 * (form[:minH, :minW] < pixel_thresh)  # yellow channel: form

        result = np.uint8(result)
        return (result)


    # pulled registration lines from specto_backend
    def register_image(self):
        pagenum =1 #TODO read template name
        n = pagenum - 1  # assume input page numbers are 1 to 6

        # load image to be registered, and apply some smoothing/deskewing
        # (following lines copied from Specto's Page class)
        im = cv2.imread(self.filenameMoving, flags=0)
        im = cv2.GaussianBlur(im, (3, 3), 0)
        im = cv2.fastNlMeansDenoising(im, h=10, templateWindowSize=7, searchWindowSize=11)
        im = np.invert(im)  # invert the images so that 0 means white

        #im = deskew(im, crop=True)#TODO has problem


        im = cv2.resize(im, dsize=self.templates[n].shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        if self.method == 'SIFT':
            homography,self.immov,points1,points2,self.keypointsM,self.descriptorsM = imReg.registration_SIFT(0,im, self.templates[n], min_match_count=4,downsample_factor=int(2 * self.PAGE_DPI / 300), kp2=self.keypoints[n],
                                                des2=self.descriptors[n],kp1=None, des1=None)
        elif self.method=='AKAZE':
            homography, self.immov, points1, points2, self.keypointsM, self.descriptorsM = imReg.registration_AKAZE(0,
                                                                                                                   im,
                                                                                                                   self.templates[
                                                                                                                       n],
                                                                                                                   downsample_factor=int(
                                                                                                                       2 * self.PAGE_DPI / 300),
                                                                                                                   kp2=
                                                                                                                   self.keypoints[
                                                                                                                       n],
                                                                                                                   des2=
                                                                                                                   self.descriptors[
                                                                                                                       n],
                                                                                                                   kp1=None,
                                                                                                                   des1=None)

        reg_im = imReg.apply_registration(im, homography)  # apply homography to image
        reg_r = imReg.reg_measure(reg_im, self.templates[n])  # measure quality of registration
        if self.method  == 'AKAZE':
            self.textscore_AKA.setText(str(reg_r))
        elif self.method =='SIFT':
            self.textscore_SIF.setText(str(reg_r))

        self.repaint()

        #Show keypoints
        img = np.invert(self.immov)
        img = cv2.drawKeypoints(img, self.keypointsM, np.array([]), (0, 0, 255))
        img = cv2.drawKeypoints(img, points1, np.array([]), (255, 0, 0))
        self.imgmm=img
        #img = cv2.resize(img, dsize=(450, 700), interpolation=cv2.INTER_LINEAR)
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        self.movingImageQI = qImg
        self.movingImageQI.scaledToWidth(width)
        self.movingImageQI.scaledToHeight(height)
        self.pixMapm = QPixmap.fromImage(self.movingImageQI)

        self.movingImageLB.setPixmap(self.pixMapm)
        self.movingImageLB.setMask(self.pixMapm.mask())

        imgf = np.invert(self.imf)
        imgf = cv2.drawKeypoints(imgf, self.keypoints[n], np.array([]), (0, 0, 255))
        imgf = cv2.drawKeypoints(imgf, points2, np.array([]), (255, 0, 0))
        #imgf = cv2.resize(imgf, dsize=(self.immov.shape[0],self.immov.shape[1]), interpolation=cv2.INTER_LINEAR)
        self.imgff = imgf
        height, width, channel = imgf.shape
        bytesPerLine = 3 * width
        qImgf = QImage(imgf, width, height, bytesPerLine, QImage.Format_RGB888)
        self.fixedImageQI = qImgf
        self.fixedImageQI.scaledToWidth(width)
        self.fixedImageQI.scaledToHeight(height)
        self.pixMapf = QPixmap.fromImage(self.fixedImageQI)

        self.fixedImageLB.setPixmap(self.pixMapf)



        # if (show_registration):
        INV_TEMPLATES = []  # for quicker display, invert the templates now to get black and white to display properly
        for temp in self.templates:
            INV_TEMPLATES.append(np.invert(temp))
        # alignment_vis = combine_images(INV_TEMPLATES[n], np.invert(reg_im))
        # generate an RGB image of the same shape as template
        pixel_thresh=127
        template = INV_TEMPLATES[n]
        form = np.invert(reg_im)
        t_shape = template.shape
        f_shape = form.shape
        minH = min(t_shape[0], f_shape[0])
        minW = min(t_shape[1], f_shape[1])

        result = np.ones((minH, minW, 3), np.uint8) * 255  # init to [255,255,255] everywhere = white image

        result[:minH, :minW, 1] = 255 - 255 * (template[:minH, :minW] < pixel_thresh)  # red channel: template
        result[:minH, :minW, 0] = 255 - 255 * (form[:minH, :minW] < pixel_thresh)  # blue channel: form
        result[:minH, :minW, 2] = 255 - 255 * (form[:minH, :minW] < pixel_thresh)  # yellow channel: form

        alignment_vis = np.uint8(result)

        alignment_vis= cv2.resize(alignment_vis, dsize=(450, 700), interpolation=cv2.INTER_LINEAR)


        height, width, channel =alignment_vis.shape
        bytesPerLine = 3 * width
        qImg = QImage(alignment_vis.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.resultQI = qImg
        pixMapR = QPixmap.fromImage(self.resultQI)
        self.resultLB.setPixmap(pixMapR)
        self.resultLB.setMask(pixMapR.mask())

    def try_registeragain(self):
        pagenum = 1  # TODO read template name
        n = pagenum - 1  # assume input page numbers are 1 to 6

        # load image to be registered, and apply some smoothing/deskewing
        # (following lines copied from Specto's Page class)
        im = cv2.imread(self.filenameMoving, flags=0)
        im = cv2.GaussianBlur(im, (3, 3), 0)
        im = cv2.fastNlMeansDenoising(im, h=10, templateWindowSize=7, searchWindowSize=11)
        im = np.invert(im)  # invert the images so that 0 means white

        # im = deskew(im, crop=True)#TODO has problem

        im = cv2.resize(im, dsize=self.templates[n].shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        if self.method=='AKAZE':
            homography, self.immov, points1, points2, self.keypointsM, self.descriptorsM = imReg.registration_AKAZE(1,im,
                                                                                                                self.templates[
                                                                                                                    n],
                                                                                                                downsample_factor=int(
                                                                                                                    2 * self.PAGE_DPI / 300),
                                                                                                                kp2=
                                                                                                                self.keypoints[
                                                                                                                    n],
                                                                                                                des2=
                                                                                                                self.descriptors[
                                                                                                                    n],
                                                                                                                kp1=self.keypointsM,
                                                                                                                des1=self.descriptorsM)
        elif self.method=='SIFT':
            homography, self.immov, points1, points2, self.keypointsM, self.descriptorsM = imReg.registration_SIFT(1,
                                                                                                                    im,
                                                                                                                    self.templates[
                                                                                                                        n],
                                                                                                                    min_match_count=4,
                                                                                                                    downsample_factor=int(
                                                                                                                        2 * self.PAGE_DPI / 300),
                                                                                                                    kp2=
                                                                                                                    self.keypoints[
                                                                                                                        n],
                                                                                                                    des2=
                                                                                                                    self.descriptors[
                                                                                                                        n],
                                                                                                                    kp1=self.keypointsM,
                                                                                                                    des1=self.descriptorsM)
        reg_im = imReg.apply_registration(im, homography)  # apply homography to image
        reg_r = imReg.reg_measure(reg_im, self.templates[n])  # measure quality of registration
        if self.method =='AKAZE':
            self.textscore_AKA.setText(str(reg_r))
        elif self.method == 'SIFT':
            self.textscore_SIF.setText(str(reg_r))
        self.repaint()

        # Show keypoints
        img = np.invert(self.immov)
        img = cv2.drawKeypoints(img, self.keypointsM, np.array([]), (0, 0, 255))
        img = cv2.drawKeypoints(img, points1, np.array([]), (255, 0, 0))
        self.imgmm = img

        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        self.movingImageQI = qImg
        self.movingImageQI.scaledToWidth(width)
        self.movingImageQI.scaledToHeight(height)
        self.pixMapm = QPixmap.fromImage(self.movingImageQI)

        self.movingImageLB.setPixmap(self.pixMapm)
        self.movingImageLB.setMask(self.pixMapm.mask())

        imgf = np.invert(self.imf)
        imgf = cv2.drawKeypoints(imgf, self.keypoints[n], np.array([]), (0, 0, 255))
        imgf = cv2.drawKeypoints(imgf, points2, np.array([]), (255, 0, 0))
        # imgf = cv2.resize(imgf, dsize=(self.immov.shape[0],self.immov.shape[1]), interpolation=cv2.INTER_LINEAR)
        self.imgff = imgf
        height, width, channel = imgf.shape
        bytesPerLine = 3 * width
        qImgf = QImage(imgf, width, height, bytesPerLine, QImage.Format_RGB888)
        self.fixedImageQI = qImgf
        self.fixedImageQI.scaledToWidth(width)
        self.fixedImageQI.scaledToHeight(height)
        self.pixMapf = QPixmap.fromImage(self.fixedImageQI)

        self.fixedImageLB.setPixmap(self.pixMapf)
        # self.scrolf.setWidget(self.fixedImageLB)
        # self.scrolf.setVisible(True)
        # self.scrolf.setWidgetResizable(True)
        # self.scrolf.setFixedHeight(700)
        # self.scrolf.setFixedWidth(450)
        # self.fixedImageLB.setMask(pixMapf.mask())

        # if (show_registration):
        INV_TEMPLATES = []  # for quicker display, invert the templates now to get black and white to display properly
        for temp in self.templates:
            INV_TEMPLATES.append(np.invert(temp))
        # alignment_vis = combine_images(INV_TEMPLATES[n], np.invert(reg_im))
        # generate an RGB image of the same shape as template
        pixel_thresh = 127
        template = INV_TEMPLATES[n]
        form = np.invert(reg_im)
        t_shape = template.shape
        f_shape = form.shape
        minH = min(t_shape[0], f_shape[0])
        minW = min(t_shape[1], f_shape[1])

        result = np.ones((minH, minW, 3), np.uint8) * 255  # init to [255,255,255] everywhere = white image

        result[:minH, :minW, 1] = 255 - 255 * (template[:minH, :minW] < pixel_thresh)  # red channel: template
        result[:minH, :minW, 0] = 255 - 255 * (form[:minH, :minW] < pixel_thresh)  # blue channel: form
        result[:minH, :minW, 2] = 255 - 255 * (form[:minH, :minW] < pixel_thresh)  # yellow channel: form

        alignment_vis = np.uint8(result)

        alignment_vis = cv2.resize(alignment_vis, dsize=(450, 700), interpolation=cv2.INTER_LINEAR)

        height, width, channel = alignment_vis.shape
        bytesPerLine = 3 * width
        qImg = QImage(alignment_vis.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.resultQI = qImg
        pixMapR = QPixmap.fromImage(self.resultQI)
        self.resultLB.setPixmap(pixMapR)
        self.resultLB.setMask(pixMapR.mask())









        #     plt.rcParams["figure.figsize"] = [15, 20]
        #     plt.imshow(alignment_vis)  # , cmap='bgr')




def main():
    app = QApplication(sys.argv)
    mygui = MyGui()



    sys.exit(app.exec_())

if __name__ == '__main__':
    main()





