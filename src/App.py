import sys
from PySide6.QtWidgets import *

class MainWindow(QMainWindow):
    def __init__ (self):
        super().__init__() 
        self.resize(900,600)
        self.setWindowTitle("App")

        self.label = QLabel()
        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)


        button = QPushButton("Hello World!")
        button.setFixedSize(100,100)
        button.setCheckable(True)
        button.clicked.connect(self.buttonClick)

        layout = QVBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.label)
        layout.addWidget(button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def buttonClick(self):
        print("I was clicked")
        

class Button:
    def __init__(self):
        super().__init__
        


app = QApplication(sys.argv)
window = MainWindow()
window.show()

sys.exit(app.exec())

