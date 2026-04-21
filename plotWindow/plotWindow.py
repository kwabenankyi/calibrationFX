import matplotlib
# prevent NoneType error for versions of matplotlib 3.1.0rc1+ by calling matplotlib.use()
# For more on why it's nececessary, see
# https://stackoverflow.com/questions/59656632/using-qt5agg-backend-with-matplotlib-3-1-2-get-backend-changes-behavior
matplotlib.use('qt5agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QEventLoop
import sys
import time

"""
Heavily extended from the original code by James Jackson, Jacob Willis, and David Pagnon.

Creators: James Jackson, Jacob Willis, David PAGNON
Authors: Anthony Nkyi
Latest edits: April 2026.

Sources: 
https://github.com/superjax/plotWindow [Retrieved 2026-04-20]
"""

class plotWindow():
    def __init__(self, parent=None):
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.setWindowTitle("plot window")
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.closeTab)
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(1280, 900)
        self.MainWindow.show()

    def closeTab(self, index):
        self.tabs.widget(index).deleteLater()
        self.tabs.removeTab(index)
        del self.toolbar_handles[index]
        del self.canvases[index]
        del self.figure_handles[index]
        del self.tab_handles[index]

    def addPlot(self, title, figure):
        new_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        new_tab.setLayout(layout)

        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)
        self.current_window = self.tabs.count() - 1
        self.tabs.setCurrentIndex(self.current_window)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

        # Ensure the newest figure is rendered and visible immediately.
        new_canvas.draw_idle()
        self.MainWindow.showNormal()
        self.MainWindow.raise_()
        self.MainWindow.activateWindow()
        self.update()

    def update(self):
        self.app.processEvents(QEventLoop.AllEvents, 25)

    def show(self):
        self.MainWindow.show()
        self.update()

    def close(self):
        self.MainWindow.close()

if __name__ == '__main__':
    pw = plotWindow()
    x = np.arange(0, 10, 0.001)

    f1 = plt.figure()
    ysin = np.sin(x)
    plt.plot(x, ysin, '--')
    pw.addPlot("sin", f1)
    pw.show()

    for _ in range(30):
        time.sleep(0.1)
        pw.update()

    f2 = plt.figure()
    ycos = np.cos(x)
    plt.plot(x, ycos, '--')
    pw.addPlot("cos", f2)

    while pw.MainWindow.isVisible():
        pw.update()
        time.sleep(0.05)
