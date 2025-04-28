import sys
import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QMouseEvent
from PyQt5.uic import loadUi
import time
import segmentador_sam
import cv2
import threading

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("untitled.ui", self)
        
        self.fname = None
        self.fname_2 = None
        self.obj_num = 0

        self.pushButton.clicked.connect(self.clickhandler)
        self.pushButton_2.clicked.connect(self.clickhandler_2)

        self.pushButton_3.clicked.connect(self.clickhandler_3)

    def clickhandler(self):
        fname = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
        self.lineEdit.setText(fname)
        if fname == "" : self.fname = None
        else : self.fname = fname

    def clickhandler_2(self):
        fname_2 = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
        self.lineEdit_2.setText(fname_2)
        self.fname_2 = fname_2
        if fname_2 == "" : self.fname_2 = None
        else : self.fname_2 = fname_2

    def clickhandler_3(self):
        
        if self.fname is None or self.fname_2 is None:
            self.print_terminal("Please set both Directories")
            return
        
        self.segment()

    def display_info(self, crop_info):
        bbox_new = [crop_info[4][0], crop_info[4][1], crop_info[4][0]+crop_info[4][2], crop_info[4][1]+crop_info[4][3]]
        self.print_terminal("")
        self.print_terminal("File Path: " + crop_info[0])
        self.print_terminal("Image from: " + crop_info[1])
        self.print_terminal("Colour: " + crop_info[2])
        self.print_terminal("Size (px): " + str(crop_info[3]))
        self.print_terminal("Coordinates: " + str(bbox_new))

        img_path = os.path.join(crop_info[5], "final_image.jpg")
        img_save = os.path.join(crop_info[5], "buffer_image.jpg")
        img = cv2.imread(img_path)

        h, w, l = img.shape
        ratio = 1024 / w
        if ratio > 1: ratio = 1
        img = cv2.resize(img, None, fx=ratio, fy=ratio)

        height, width, channel = img.shape
        thickness = int(max(height, width)/500)
        img = cv2.rectangle(img, (int(bbox_new[0]), int(bbox_new[1])), (int(bbox_new[2]), int(bbox_new[3])), (0,0,0), thickness)
        cv2.imwrite(img_save, img)
        
        img_pixmap = QPixmap(img_save)
        self.label_4.setScaledContents(True)
        self.label_4.setPixmap(img_pixmap)

    def segment(self):
        self.print_terminal("Segmentador-Clasificador Inicializado!")
        self.print_terminal("")

        self.progressBar.reset()
        self.progressBar.setEnabled(True)
        self.progressBar.setTextVisible(True)
        self.pushButton_3.setEnabled(False)
        self.lineEdit_3.setEnabled(False)

        self.thread = Thread()
        self.thread.new_image.connect(self.add_image)
        self.thread.new_terminal_msg.connect(self.print_terminal)
        self.thread.progress_max.connect(self.set_progress_max)
        self.thread.new_progress_value.connect(self.add_progress_bar)
        self.thread.start()

    def print_terminal(self, text):
        self.textBrowser.append(text)

    def add_image(self, img_array):
        self.new_image = QClickableLabel(self.scrollAreaWidgetContents)
        self.new_image.info = img_array
        self.new_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.new_answer = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.new_label.setText(img_array[1])
        self.new_answer.setText(img_array[2])

        #sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #sizePolicy.setHeightForWidth(self.new_label.sizePolicy().hasHeightForWidth())
        #self.new_label.setSizePolicy(sizePolicy)
        self.new_image.setMinimumSize(QtCore.QSize(50,50))
        self.new_image.setMaximumSize(QtCore.QSize(50,50))
        self.new_image.setObjectName("label" + str(self.obj_num))
        pixmap = QPixmap(img_array[0])
        self.new_image.setScaledContents(True)
        self.new_image.setPixmap(pixmap)
        self.new_image.clicked.connect(self.display_info)
        self.new_label.setMaximumSize(QtCore.QSize(50,50))

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.new_image)
        self.verticalLayout.addWidget(self.new_label)
        self.verticalLayout.addWidget(self.new_answer)
        self.gridLayout.addLayout(self.verticalLayout, int(self.obj_num/5), (self.obj_num%5), 1, 1)
        self.obj_num = self.obj_num + 1

    def set_progress_max(self, value):
        self.progressBar.setMaximum(value)

    def add_progress_bar(self, value):
        self.progressBar.setValue(value)
        if value == self.progressBar.maximum():
            self.progressBar.setFormat(" %p% - " + str(value) + "/" + str(self.progressBar.maximum()) + " Complete!")
        else:
            self.progressBar.setFormat(" %p% - " + str(value) + "/" + str(self.progressBar.maximum()) + "...")


class Thread(QThread):
    new_image = pyqtSignal(list)
    new_terminal_msg = pyqtSignal(str)
    progress_max = pyqtSignal(int)
    new_progress_value = pyqtSignal(int)
    def __init__(self):
        super(Thread, self).__init__()

    def run(self):
        self.new_terminal_msg.emit("Loading Segmenter/Classifier...")

        wb = segmentador_sam.WhiteBalancer(3, 0.99)
        overlap_function = segmentador_sam.OverlapFunction(3)
        separate_blobs = segmentador_sam.SeparateBlobs()
        rescaler = segmentador_sam.Rescaler(3, downscale_size=1024)
        points_per_side = int(mainwindow.lineEdit_3.text())
        i_directory = mainwindow.fname
        o_directory = mainwindow.fname_2
        
        self.new_terminal_msg.emit("PPS: " + str(points_per_side))

        seg = segmentador_sam.Segmentador(1, whitebalancer=wb, overlap_function=overlap_function, separate_blobs=separate_blobs, points_per_side=points_per_side, rescaler=rescaler)
        clas = segmentador_sam.Clasificador("What colour is the stone?")
        segclas = segmentador_sam.SegmentadorClasificador(i_directory, seg, clas, output=o_directory)

        all_results = {}
        all_results['area'] = []
        all_results['answers'] = []
        all_results['filename'] = []
        all_results['image_name'] = []
        all_results['crop'] = []
        all_results['seg_time'] = 0
        all_results['class_time'] = 0

        self.progress_max.emit(len(segclas.image_list[0]))
        self.new_progress_value.emit(0)
        for i in range(0, len(segclas.image_list[0])):
            self.new_terminal_msg.emit(str(i) + " of " + str(len(segclas.image_list[0])) + " Images inspected")
            self.new_terminal_msg.emit("Analyzing " + segclas.image_list[0][i] + "...")
            
            self.new_terminal_msg.emit("Segmenting...")
            start_t = time.time()
            segmentation_result = segclas.segmentador.segment(segclas.image_list[1][i])
            seg_t = time.time() - start_t
            image_to_crop = segmentation_result['image']
            if segmentation_result['image_ratio'] > 1: image_to_crop = segclas.image_list[1][i]
            segmentation_result = segclas.parse_seg_results(segmentation_result, image_to_crop, segmentation_result['image_ratio'])

            self.new_terminal_msg.emit("Classifying...")
            start_t = time.time()
            segmentation_result['answers_list'] = segclas.clasificador.classify(segmentation_result['array_of_plastics_crop'])
            class_t = time.time() - start_t

            image_stats = {}
            image_stats['area'] = segmentation_result['area_array']
            image_stats['answers'] = segmentation_result['answers_list']
            image_stats['bbox'] = segmentation_result['bbox_array']
            image_stats['seg_time'] = seg_t
            image_stats['class_time'] = class_t
            image_stats['crop'] = segmentation_result['array_of_plastics_crop']
            all_results['area'] = all_results['area'] + image_stats['area']
            all_results['answers'] = all_results['answers'] + image_stats['answers']
            all_results['crop'] = all_results['crop'] + image_stats['crop']
            all_results['seg_time'] = all_results['seg_time'] + seg_t
            all_results['class_time'] = all_results['class_time'] + class_t
            for k in range(0, len(image_stats['area'])):
                all_results['image_name'].append(segclas.image_list[0][i])
                all_results['filename'].append("crop" + str(k) + ".jpg")
                

            #print(segmentation_result['answers_list'])
            file_dir = os.path.join(segclas.directory, segclas.image_list[0][i])
            classify_results = segclas.parse_classify_results(file_dir, segmentation_result['image'], segmentation_result['masks_array'], segmentation_result['bbox_array'], segmentation_result['answers_list'], segmentation_result['area_array'])
            segclas.to_folder(segclas.image_list[0][i], segmentation_result['image'], segmentation_result['annotated_image'], segmentation_result['annotated_frame'], 
                           segmentation_result['masks_array'], segmentation_result['array_of_plastics_crop'], classify_results['boxed_image'])
            segclas.write_text_file(segmentador_sam.SegmentadorClasificador.crop_filename(segclas.image_list[0][i]), image_stats)
            
            self.new_terminal_msg.emit("Finalized!")
            for k in range(0, len(image_stats['area'])):
                file_dir = os.path.join(segclas.results_dir, segmentador_sam.SegmentadorClasificador.crop_filename(segclas.image_list[0][i]))
                crop_path = os.path.join(file_dir, "crop" + str(k) + ".jpg")

                img_array = []
                img_array.append(crop_path)
                img_array.append(segclas.image_list[0][i])
                img_array.append(image_stats['answers'][k][0])
                img_array.append(image_stats['area'][k])
                img_array.append(image_stats['bbox'][k])
                img_array.append(file_dir)

                self.new_image.emit(img_array)

            self.new_progress_value.emit(i+1)

        segclas.write_text_file("total", all_results)
        segclas.write_csv_file("results", all_results)

        self.new_terminal_msg.emit("")
        self.new_terminal_msg.emit("Process finalized!")

        self.exit(0)

class QClickableLabel(QLabel):
  clicked=pyqtSignal(list)
  info=[]

  def mousePressEvent(self, ev):
    self.clicked.emit(self.info)


app=QApplication(sys.argv)
mainwindow = MainWindow()
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(1200)
widget.setFixedHeight(700)
widget.setWindowTitle("Microplastic Segmenter")
widget.show()
sys.exit(app.exec_())