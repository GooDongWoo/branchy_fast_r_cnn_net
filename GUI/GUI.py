import cv2
import numpy as np
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QPushButton, QApplication, QComboBox, QLabel, QFileDialog, QStatusBar, QDesktopWidget, QMessageBox, QMainWindow

import pyqtgraph as pg
import sys
from process import Process
from webcam import Webcam
from video import Video
from interface import waitKey

class GUI(QMainWindow, QThread):
    def __init__(self):
        super(GUI,self).__init__()
        self.initUI()
        self.webcam = Webcam()
        self.video = Video()
        self.input = self.webcam
        self.dirname = ""
        print("Input: webcam")
        self.statusBar.showMessage("Input: webcam",5000)
        self.btnOpen.setEnabled(False)
        self.process = Process()
        self.status = False
        self.frame = np.zeros((10,10,3),np.uint8)
        self.terminate = False
        
    def initUI(self):
    
        #set font
        font = QFont()
        font.setPointSize(16)
        
        #widgets
        self.btnStart = QPushButton("Start", self)
        self.btnStart.move(440,520)
        self.btnStart.setFixedWidth(200)
        self.btnStart.setFixedHeight(50)
        self.btnStart.setFont(font)
        self.btnStart.clicked.connect(self.run)
        
        self.btnOpen = QPushButton("Open", self)
        self.btnOpen.move(230,520)
        self.btnOpen.setFixedWidth(200)
        self.btnOpen.setFixedHeight(50)
        self.btnOpen.setFont(font)
        self.btnOpen.clicked.connect(self.openFileDialog)
        
        self.cbbInput = QComboBox(self)
        self.cbbInput.addItem("Webcam")
        self.cbbInput.addItem("Video")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setFixedWidth(200)
        self.cbbInput.setFixedHeight(50)
        self.cbbInput.move(20,520)
        self.cbbInput.setFont(font)
        self.cbbInput.activated.connect(self.selectInput)
        #-------------------

        self.H = 360
        self.W = 480
        
        self.lblDisplay1 = QLabel(self) #label to show frame from camera
        self.lblDisplay1.setGeometry(10, 70, self.W, self.H)
        self.lblDisplay1.setStyleSheet("background-color: #000000")

        self.lblDisplay2 = QLabel(self)  # label to show frame from camera
        self.lblDisplay2.setGeometry(20 + self.W, 70, self.W, self.H)
        self.lblDisplay2.setStyleSheet("background-color: #000000")

        self.lblDisplay3 = QLabel(self)  # label to show frame from camera
        self.lblDisplay3.setGeometry(30 + self.W * 2, 70, self.W, self.H)
        self.lblDisplay3.setStyleSheet("background-color: #000000")
        
        self.lblROI = QLabel(self) #label to show face with ROIs
        self.lblROI.setGeometry(40 + self.W * 3, 130, 320, 240)
        self.lblROI.setStyleSheet("background-color: #000000")

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)

        #config main window
        self.setGeometry(30, 30, 50 + self.W * 3 + 320, 640)
        #self.center()
        self.setWindowTitle("BranchyNet-Faster RCNN")
        self.show()
        
        
    def update(self):
        # self.signal_Plt.clear()
        # self.signal_Plt.plot(self.process.samples[20:],pen='g')

        # self.fft_Plt.clear()
        # self.fft_Plt.plot(np.column_stack((self.process.freqs, self.process.fft)), pen = 'g')
        return 0

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self,"Message", "Are you sure want to quit",
            QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            self.input.stop()
            # cv2.destroyAllWindows()
            self.terminate = True
            sys.exit()

        else: 
            event.ignore()
    
    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.input = self.webcam
            print("Input: webcam")
            self.btnOpen.setEnabled(False)
            #self.statusBar.showMessage("Input: webcam",5000)
        elif self.cbbInput.currentIndex() == 1:
            self.input = self.video
            print("Input: video")
            self.btnOpen.setEnabled(True)
    
    def key_handler(self):
        """
        cv2 window must be focused for keypresses to be detected.
        """
        self.pressed = waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("[INFO] Exiting")
            self.webcam.stop()
            sys.exit()
    
    def openFileDialog(self):
        self.dirname = QFileDialog.getOpenFileName(self, 'OpenFile')
        #self.statusBar.showMessage("File name: " + self.dirname,5000)
    
    def reset(self):
        self.process.reset()
        self.lblDisplay1.clear()
        self.lblDisplay1.setStyleSheet("background-color: #000000")

    def main_loop(self):
        frame = self.input.get_frame()
        ret = False
        self.process.frame_in = frame
        if self.terminate == False:
            ret = self.process.run()

        if ret: # Last Layer
            self.frame = self.process.frame_out #get the frame to show in GUI
            self.f_fr2 = self.process.frame_ROI1 #get the face to show in GUI
            self.f_fr = self.process.frame_ROI2
            self.f_fr3 = self.process.frame_ROI3

            self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
            self.f_fr = cv2.resize(self.f_fr, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            cv2.putText(self.f_fr, "Last-Layer Faster-RCNN ", (20, self.H - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            f_img = QImage(self.f_fr, self.f_fr.shape[1], self.f_fr.shape[0],
                           self.f_fr.strides[0], QImage.Format_RGB888)
            self.lblDisplay3.setPixmap(QPixmap.fromImage(f_img))

            self.f_fr3 = cv2.cvtColor(self.f_fr3, cv2.COLOR_RGB2BGR)
            self.f_fr3 = cv2.resize(self.f_fr3, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            cv2.putText(self.f_fr3, "Adaptive-Algorithm Faster-RCNN ", (20, self.H - 20), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (0, 255, 255), 2)
            f_img3 = QImage(self.f_fr3, self.f_fr3.shape[1], self.f_fr3.shape[0],
                           self.f_fr3.strides[0], QImage.Format_RGB888)
            self.lblDisplay2.setPixmap(QPixmap.fromImage(f_img3))

            self.f_fr2 = cv2.cvtColor(self.f_fr2, cv2.COLOR_RGB2BGR)
            self.f_fr2 = cv2.resize(self.f_fr2, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
            cv2.putText(self.f_fr2, "Early-Exit Faster-RCNN ", (20, self.H - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            f_img2 = QImage(self.f_fr2, self.f_fr2.shape[1], self.f_fr2.shape[0],
                            self.f_fr2.strides[0], QImage.Format_RGB888)
            self.lblDisplay1.setPixmap(QPixmap.fromImage(f_img2))
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        self.frame = cv2.resize(self.frame, (320, 240), interpolation=cv2.INTER_CUBIC)
        cv2.putText(self.frame, "Real-Time_Video ", (20,220), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255),2)
        img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0],
                        self.frame.strides[0], QImage.Format_RGB888)
        self.lblROI.setPixmap(QPixmap.fromImage(img))
        self.key_handler()  #if not the GUI cant show anything

    def run(self, input):
        print("run")
        self.reset()
        input = self.input
        self.input.dirname = self.dirname
        if self.input.dirname == "" and self.input == self.video:
            print("choose a video first")
            #self.statusBar.showMessage("choose a video first",5000)
            return
        if self.status == False:
            self.status = True
            input.start()
            self.btnStart.setText("Stop")
            self.cbbInput.setEnabled(False)
            self.btnOpen.setEnabled(False)
            # self.lblHR2.clear()
            while self.status == True:
                self.main_loop()

        elif self.status == True:
            self.status = False
            input.stop()
            self.btnStart.setText("Start")
            self.cbbInput.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
