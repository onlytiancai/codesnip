import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.label = QLabel(self)
        self.label.setText('Hello, World!')
        self.label.move(50, 50)

        closeButton = QPushButton('Close', self)
        closeButton.move(100, 100)
        closeButton.clicked.connect(self.close)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowFlags(Qt.WindowStaysOnTopHint|Qt.FramelessWindowHint)
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 标记是否按下
            self.m_flag = True
            # 获取鼠标相对窗口的位置
            self.m_Position = event.globalPos() - self.pos()
            event.accept()

    def mouseMoveEvent(self, QMouseEvent):
        try:
            # 仅监听标题栏
            if Qt.LeftButton and self.m_flag:
                # 更改鼠标图标
                self.setCursor(QCursor(Qt.OpenHandCursor))
                # 更改窗口位置
                self.move(QMouseEvent.globalPos() - self.m_Position)
                QMouseEvent.accept()
        except Exception as e:
            print("报错信息=", e)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # 恢复鼠标形状
        self.setCursor(QCursor(Qt.ArrowCursor))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MyWidget()
    sys.exit(app.exec_())
