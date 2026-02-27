import sys
from tkinter.font import Font
from PySide6.QtWidgets import *
from PySide6.QtWidgets import QHeaderView, QAbstractItemView
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QPalette
class MainWindow(QMainWindow):
    def __init__ (self):
        super().__init__() 
        self.resize(900,600)
        self.setWindowTitle("App")
        self.windowTracker = []
        self.setStyleSheet('background-color: #C0C0C0')
        

        
        self.AllWindows = QStackedWidget() #Basically a container that keeps all the windows (widgets) for the application
        mainPageLayout = QHBoxLayout() #The layout for this page MainWindow.
        self.mainPage = QWidget() #The page in itself.

        #Definition of different pages for the application:
        self.filePage = FilePage()
        self.creationPage = CreatePage()

        #Definition of buttons that will redirect.
        fileButton = QPushButton("Go to files")
        fileButton.setStyleSheet('background-color: grey')
        createButton = QPushButton("Start creating")
        createButton.setStyleSheet('background-color: grey')
        self.backButton = QPushButton("Go Back")
        self.backButton.setStyleSheet('background-color: grey')

        createButton.setFixedSize(100,100)
        fileButton.setFixedSize(100,100)
        self.backButton.setFixedSize(80,50)

        #Connects the buttons the method that handles redirection
        fileButton.clicked.connect(self.goToFilePage)
        createButton.clicked.connect(self.goToCreationPage)
        self.backButton.clicked.connect(self.goBack)
        
        #Add the buttons to the layout
        mainPageLayout.addWidget(createButton)
        mainPageLayout.addWidget(fileButton)
        

        #Put the page layout inside the mainpage.
        self.mainPage.setLayout(mainPageLayout)


        #Now add all the pages into the container for all the windows in the application.
        self.AllWindows.addWidget(self.mainPage)
        self.AllWindows.addWidget(self.filePage)
        self.AllWindows.addWidget(self.creationPage)

        #Center the container, and make the current page the mainpage.
        self.setMenuWidget(self.backButton)
        self.backButton.hide()
        self.setCentralWidget(self.AllWindows)
        self.AllWindows.setCurrentWidget(self.mainPage)


    def goToFilePage(self):
        self.backButton.show()
        self.windowTracker.append(self.mainPage)
        self.AllWindows.setCurrentWidget(self.filePage)

    def goToCreationPage(self):
        self.backButton.show()
        self.windowTracker.append(self.mainPage)
        self.AllWindows.setCurrentWidget(self.creationPage)
    
    def goBack(self):
        currentWindow = self.windowTracker.pop()
        if(len(self.windowTracker) == 0):
            self.backButton.hide()
        self.AllWindows.setCurrentWidget(currentWindow)

class FilePage(QWidget):
    def __init__ (self):
        super().__init__()
        
        label = QLabel()
        label.setText("Welcome to the file page")
        font = QFont('', 24)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        Table = QTableWidget(10, 5)
        Table.setStyleSheet('background-color: grey')
        Table.setHorizontalHeaderLabels(["Name", "Created", "Size", "Type", "Download"])
        Table.setSizePolicy(QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum))
        Table.setFixedHeight(300)

        header = Table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(Table)
        center = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        layout.setAlignment(center)

        self.setLayout(layout)





        


class CreatePage(QWidget):
    def __init__ (self):
        super().__init__()
        label = QLabel()
        label.setText("Welcome to the Creation page")
        font = QFont('', 24)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)



        

app = QApplication(sys.argv)
QTimer.singleShot(30_000, app.quit)
window = MainWindow()
window.show()

sys.exit(app.exec())

