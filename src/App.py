import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import QTimer
class MainWindow(QMainWindow):
    def __init__ (self):
        super().__init__() 
        self.resize(900,600)
        self.setWindowTitle("App")

        
        self.AllWindows = QStackedWidget() #Basically a container that keeps all the windows (widgets) for the application
        mainPageLayout = QHBoxLayout() #The layout for this page MainWindow.
        mainPage = QWidget() #The page in itself.

        #Definition of different pages for the application:
        self.filePage = FilePage()
        self.creationPage = CreatePage()

        #Definition of buttons that will redirect.
        fileButton = QPushButton("Go to files")
        createButton = QPushButton("Start creating")
        createButton.setFixedSize(100,100)
        createButton.setCheckable(True)
        fileButton.setFixedSize(100,100)
        fileButton.setCheckable(True)

        #Connects the buttons the method that handles redirection
        fileButton.clicked.connect(self.goToFilePage)
        createButton.clicked.connect(self.goToCreationPage)
        
        #Add the buttons to the layout
        mainPageLayout.addWidget(createButton)
        mainPageLayout.addWidget(fileButton)

        #Put the page layout inside the mainpage.
        mainPage.setLayout(mainPageLayout)

        #Now add all the pages into the container for all the windows in the application.
        self.AllWindows.addWidget(mainPage)
        self.AllWindows.addWidget(self.filePage)
        self.AllWindows.addWidget(self.creationPage)

        #Center the container, and make the current page the mainpage.
        self.setCentralWidget(self.AllWindows)
        self.AllWindows.setCurrentWidget(mainPage)

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

