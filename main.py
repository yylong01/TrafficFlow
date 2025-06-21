import time

from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtUiTools import QUiLoader
import cv2 as cv
from PySide6.QtCore import QTimer, Qt, Signal, QObject
from PySide6.QtGui import QImage, QPixmap

from ultralytics import YOLO
from threading import Thread



class SignalMar(QObject):
    imgGen = Signal(QPixmap)
    carCount = Signal(str)

sm = SignalMar()

class VideoHandler:
    def __init__(self):
        # 定义定时器，用于控制显示视频文件的帧率
        self.timer_videoFile = QTimer()
        # 定时到了，回调self.show_videoFile
        self.timer_videoFile.timeout.connect(self.play_video_file)
        # cv2.videoCapture 实例
        self.cap = None
        self.frameToPredict = None
        self.car_count = 0
        self.plainTextEdit = None

        # 加载模型
        self.model = YOLO("best.pt")

        sm.imgGen.connect(self.showImgGen)
        sm.carCount.connect(self.addCarCount)

        # 启动YoLo处理视频帧的独立线程
        thread = Thread(target=self.threadFunc_frmaePredict, daemon=True)
        thread.start()

    def set_video_label(self, label, label2, plainTextEdit):
        self.label_ori = label
        self.label_gen = label2
        self.plainTextEdit = plainTextEdit

    def openVideoFile(self, win):
        videoPath, _ = QFileDialog.getOpenFileName(
            win,    # 父窗口
            "请选择视频文件", # 标题
            r".",   # 起始目录
            "视频类型(*.mp4 *.avi)"
        )

        if not videoPath:
            return
        self.cap = cv.VideoCapture(videoPath)
        if not self.cap.isOpened():
            print("打开视频文件失败")
            return

        self.timer_videoFile.start(30)

    def play_video_file(self):
        ret, frame = self.cap.read()

        if not ret:
            print("视频播放结束，重新开始")
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        # 视频色彩换回RGB
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        qImage = QImage(frame.data, frame.shape[1], frame.shape[0],
                        QImage.Format_RGB888)
        # 往显示视频的Label里显示QImage
        pixmap = QPixmap.fromImage(qImage)
        scaled_pix = pixmap.scaled(
            self.label_ori.size(),
            aspectMode=Qt.KeepAspectRatio
        )
        self.label_ori.setPixmap(scaled_pix)

        # 当前后台YoLo线程没有处理任务
        if self.frameToPredict is None:
            self.frameToPredict = frame



    def threadFunc_frmaePredict(self):
        while True:
            # 视频2

            if self.frameToPredict is None:
                time.sleep(0.01)
                continue

            result = self.model.predict(self.frameToPredict)[0]

            names = result.names
            cls_indexes = result.boxes.cls.cpu().numpy().astype(int)
            cls_names = [names[i] for i in cls_indexes]
            self.car_count = str(cls_names.count('car'))
            # self.plainTextEdit.appendPlainText("当前通过车辆:" + self.car_count)
            # 不要再主线程外处理ui界面，故采用信号的方式
            sm.carCount.emit("当前通过车辆:" + self.car_count)


            img = result.plot(line_width=1)
            qImage2 = QImage(img.data, img.shape[1], img.shape[0],
                             QImage.Format_RGB888)
            pixmap2 = QPixmap.fromImage(qImage2)
            scaled_pix2 = pixmap2.scaled(
                self.label_gen.size(),
                aspectMode=Qt.KeepAspectRatio
            )

            # 不要再主线程外处理ui界面，故采用信号的方式
            sm.imgGen.emit(scaled_pix2)


            # 没有必要实时预测
            time.sleep(1)
            self.frameToPredict = None
    def showImgGen(self, img):
        self.label_gen.setPixmap(img)
    def addCarCount(self, info):
        self.plainTextEdit.appendPlainText("当前通过车辆:" + self.car_count)


    def stop(self):
        self.frameToPredict = None
        if self.cap:
            self.cap.release()

        self.label_ori.clear()
        self.label_gen.clear()
        self.timer_videoFile.stop()


vh = VideoHandler()

class mainWin:
    def __init__(self):
        loader = QUiLoader()
        ui_path = "ui/mainWin.ui"
        self.ui = loader.load(ui_path)
        self.ui.show()



        # 点击视频文件
        self.ui.pushButton.clicked.connect(lambda :vh.openVideoFile(self.ui))
        vh.set_video_label(self.ui.label, self.ui.label_2, self.ui.plainTextEdit)

        self.ui.pushButton_3.clicked.connect(vh.stop)







if __name__ == '__main__':
    app = QApplication()
    win = mainWin()
    app.exec()