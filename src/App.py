import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import QTimer
class MainWindow(QMainWindow):
    def __init__ (self):
        super().__init__() 
        self.resize(900,600)
        self.setWindowTitle("App")

        self.AllWindows = QStackedWidget()
        mainPageLayout = QHBoxLayout()
        mainPage = QWidget()
        self.filePage = FilePage()
        self.creationPage = CreatePage()

        fileButton = QPushButton("Go to files")
        createButton = QPushButton("Start creating")
        createButton.setFixedSize(100,100)
        createButton.setCheckable(True)
        fileButton.setFixedSize(100,100)
        fileButton.setCheckable(True)

        fileButton.clicked.connect(self.goToFilePage)
        createButton.clicked.connect(self.goToCreationPage)
        
        mainPageLayout.addWidget(createButton)
        mainPageLayout.addWidget(fileButton)

        mainPage.setLayout(mainPageLayout)

        self.AllWindows.addWidget(mainPage)
        self.AllWindows.addWidget(self.filePage)
        self.AllWindows.addWidget(self.creationPage)

        self.AllWindows.setCurrentWidget(mainPage)
        self.setCentralWidget(self.AllWindows)

    def goToFilePage(self):
        self.AllWindows.setCurrentWidget(self.filePage)

    def goToCreationPage(self):
        self.AllWindows.setCurrentWidget(self.creationPage)
        

class FilePage(QWidget):
    def __init__ (self):
        super().__init__()
        label = QLabel()
        label.setText("Welcome to the file page")
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)

        


class CreatePage(QWidget):
    def __init__ (self):
        super().__init__()
        label = QLabel()
        label.setText("Welcome to the Creation page")
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)



        

app = QApplication(sys.argv)
QTimer.singleShot(30_000, app.quit)
window = MainWindow()
window.show()

sys.exit(app.exec())

