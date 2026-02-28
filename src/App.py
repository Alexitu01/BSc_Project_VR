import sys
from tkinter.font import Font
from PySide6.QtWidgets import *
from PySide6.QtWidgets import QHeaderView, QAbstractItemView
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QPalette
class MainWindow(QMainWindow):
    def __init__ (self):
        super().__init__() 

        #Defines the relative size, name and color of the app window.
        self.resize(900,600)
        self.setWindowTitle("App")
        self.setStyleSheet('background-color: #C0C0C0')

        #Stack to keep track of the 'endpoints' the user traverses to.
        self.windowTracker = []


        
        self.AllWindows = QStackedWidget() #Basically a container that keeps all the windows (widgets) for the application
        mainPageLayout = QHBoxLayout() #The layout for this page MainWindow.
        self.mainPage = QWidget() #The page in itself.
        self.current = self.mainPage

        #Definition of different pages for the application:
        self.filePage = FilePage()
        self.creationPage = CreatePage(self.goTo)
        self.templates = TemplateClass()
        self.Ai = AiClass()
        
        #Defines a dictionary to map strings to the instantiated pages.
        self.dictionary = {
            "create": self.creationPage,
            "files": self.filePage,
            "templates": self.templates,
            "ai": self.Ai,
        }

        #Definition of buttons that will redirect.
        fileButton = QPushButton("Go to files")
        createButton = QPushButton("Start creating")
        self.backButton = QPushButton("Go Back")
        
        #Set style and size of the buttons
        fileButton.setStyleSheet('background-color: grey')
        fileButton.setFixedSize(100,100)
        createButton.setStyleSheet('background-color: grey')
        createButton.setFixedSize(100,100)
        self.backButton.setStyleSheet('background-color: grey')
        self.backButton.setFixedSize(80,50)

        

        #Connects the buttons the method that handles redirection
        fileButton.clicked.connect(lambda:self.goTo("files"))
        createButton.clicked.connect(lambda:self.goTo("create"))
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
        self.AllWindows.addWidget(self.templates)
        self.AllWindows.addWidget(self.Ai)

        #Center the container, and make the current page the mainpage.
        self.setMenuWidget(self.backButton)
        self.backButton.hide()
        self.setCentralWidget(self.AllWindows)
        self.AllWindows.setCurrentWidget(self.mainPage)

#Method for navigating to different pages.
#Works by making the back-button visible (not the smartest way of doing it)
#Append the current window to stack + change the current page to where the user is navigating to.
    def goTo(self, page):
        self.backButton.show()
        self.windowTracker.append(self.current)
        self.current = self.dictionary.get(page)
        self.AllWindows.setCurrentWidget(self.current)
    
#Method for going back through the stack
#Works by popping from the stack, and navigating to that popped page while updating the current page to that page.
    def goBack(self):
        currentWindow = self.windowTracker.pop()
        if(len(self.windowTracker) == 0):
            self.backButton.hide()
        self.current = currentWindow
        self.AllWindows.setCurrentWidget(currentWindow)

class FilePage(QWidget):
    def __init__ (self):
        super().__init__()
        
        label = QLabel()
        label.setText("Welcome to the file page")
        font = QFont('', 24)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter) #Centers

        #Table for containing - in the future should be built from amount of files in a folder.
        Table = QTableWidget(10, 5)
        Table.setStyleSheet('background-color: grey')
        Table.setHorizontalHeaderLabels(["Name", "Created", "Size", "Type", "Download"]) #Set header values
        Table.setSizePolicy(QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)) #Some weird stuff about the table's size.
        Table.setFixedHeight(300)

        #Make the headers stretch to fit.
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
    def __init__ (self, goTo):
        super().__init__()
        label = QLabel()
        label.setText("Welcome to the Creation page")
        font = QFont('', 24)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        

        fromscratchButton = QPushButton("Create from Scratch")
        fromtempButton = QPushButton("Create from Templates")
        fromscratchButton.setStyleSheet('background-color: grey')
        fromtempButton.setStyleSheet('background-color: grey')
        

        fromscratchButton.clicked.connect(lambda:goTo("ai"))
        fromtempButton.clicked.connect(lambda:goTo("templates"))


        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(fromscratchButton)
        layout.addWidget(fromtempButton)
        self.setLayout(layout)




class TemplateClass(QWidget):
    def __init__ (self):
        super().__init__()
        label = QLabel()
        label.setText("Welcome to the Template Page")
        font = QFont('', 24)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)


class AiClass(QWidget):
    def __init__ (self):
        super().__init__()
        label = QLabel()
        label.setText("Welcome to the Ai Page")
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

