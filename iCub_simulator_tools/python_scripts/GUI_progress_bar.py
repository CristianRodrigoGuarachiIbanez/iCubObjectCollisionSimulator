from PyQt5 import QtGui, QtWidgets
from PyQt5 import QtCore
import sys
import time

from PyQt5.QtWidgets import (QApplication, QDialog, QProgressBar, QPushButton)

TIME_LIMIT = 100


class Actions(QDialog):
    """
    Simple dialog that consists of a Progress Bar and a Button.
    Clicking on the button results in the start of a timer and
    updates the progress bar.
    """

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self) -> None:
        self.setWindowTitle('Progress Bar')
        self.progress: QProgressBar = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        self.button: QPushButton = QPushButton('Start', self)
        self.button.move(0, 30)
        self.show()

        self.button.clicked.connect(self.__onButtonClick)

    def __onButtonClick(self) -> None:
        count = 0
        while count < TIME_LIMIT:
            count += 1
            time.sleep(1)
            self.progress.setValue(count)


if __name__ == "__main__":
    app: QApplication = QApplication(sys.argv)
    window: Actions = Actions()
    sys.exit(app.exec_())

