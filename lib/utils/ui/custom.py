from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph import PlotItem

from .ui import Curve_Menu_Form


class CustomPlotItem(PlotItem):
    signal_with_id = pyqtSignal(list)

    def __init__(self, **kwargs):
        PlotItem.__init__(self, **kwargs)

        self.id = 1
        # self.create_custom_menu()

    def setID(self, id):
        self.id = id

    def handle_custom_menu_sig(self, custom_args):
        if isinstance(custom_args, list):
            custom_args.append(self.id)
            self.signal_with_id.emit(custom_args)

    def create_custom_menu(self, num_checkboxes):
        myWidget = QWidget()
        self.custom_menu = Curve_Menu_Form()
        self.custom_menu.setupUi(myWidget, num_checkboxes)

        myMenu = QMenu(QCoreApplication.translate('CustomPlotItem', 'Show'))
        act = QWidgetAction(self)
        act.setDefaultWidget(myWidget)
        myMenu.addAction(act)

        self.subMenus.append(myMenu)
        self.ctrlMenu.addMenu(myMenu)
        # self.stateGroup.autoAdd(myMenu)
        self.custom_menu.check_sig.connect(self.handle_custom_menu_sig)

    def remove_custom_menu(self):
        myMenu = self.subMenus.pop()
        self.ctrlMenu.removeAction(myMenu.menuAction())


class Plot3dToolBar(QObject):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName('Form')
        Form.resize(718, 425)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.toolButton = QToolButton(Form)
        self.toolButton.setObjectName('toolButton')
        self.toolButton.setGeometry(QRect(60, 80, 20, 240))
        self.toolButton.setStyleSheet(
            'QToolButton{\n'
            '   background-color: rgb(133, 218, 209);\n'
            '   border-top-style: none;\n'
            '   border-top-right-radius: 10px;\n'
            '   border-bottom-right-radius: 10px;\n'
            '   border-left-style: solid;\n'
            '   border-left-color: black;\n'
            '   border-left-width: 1px;\n'
            '   border-bottom-style: none;\n'
            '   border-right-style: none;\n'
            '}\n'
            'QToolButton:hover{    \n'
            '   background-color: rgb(90, 218, 205);\n'
            '}'
        )
        self.toolButton.setCheckable(True)
        self.toolButton.setArrowType(Qt.LeftArrow)
        self.scrollArea = QScrollArea(Form)
        self.scrollArea.setObjectName('scrollArea')
        self.scrollArea.setGeometry(QRect(0, 80, 60, 240))
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setLineWidth(0)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName('scrollAreaWidgetContents')
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 60, 240))
        self.scrollAreaWidgetContents.setStyleSheet('background-color: rgb(0, 170, 255);')
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName('verticalLayout')
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton_1 = QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_1.setObjectName('pushButton_1')
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pushButton_1.sizePolicy().hasHeightForWidth())
        self.pushButton_1.setSizePolicy(sizePolicy1)
        self.pushButton_1.setMinimumSize(QSize(60, 60))
        self.pushButton_1.setStyleSheet(
            'QPushButton{\n'
            '   background-color: rgb(70, 70, 70);\n'
            '   border-top-style: none;\n'
            '   border-top-right-radius: 10px;\n'
            '   border-bottom-style: solid;\n'
            '   border-bottom-color: black;\n'
            '   border-bottom-width: 1px;\n'
            '   border-left-style: none;\n'
            '   border-right-style: none;\n'
            '}\n'
            'QPushButton::checked{\n'
            '   background-color: rgb(255, 170, 0);\n'
            '}\n'
            'QPushButton:hover:!checked{\n'
            '   background-color: rgb(100, 100, 100);\n'
            '}'
        )
        self.pushButton_1.setCheckable(True)
        self.pushButton_1.setAutoExclusive(False)

        self.verticalLayout.addWidget(self.pushButton_1)

        self.pushButton_2 = QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_2.setObjectName('pushButton_2')
        sizePolicy1.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy1)
        self.pushButton_2.setMinimumSize(QSize(60, 60))
        self.pushButton_2.setStyleSheet(
            'QPushButton{\n'
            '   background-color: rgb(70, 70, 70);\n'
            '   border-top-style: none;\n'
            '   border-bottom-style: solid;\n'
            '   border-bottom-color: black;\n'
            '   border-bottom-width: 1px;\n'
            '   border-left-style: none;\n'
            '   border-right-style: none;\n'
            '}\n'
            'QPushButton::checked{\n'
            '   background-color: rgb(255, 170, 0);\n'
            '}\n'
            'QPushButton:hover:!checked{\n'
            '   background-color: rgb(100, 100, 100);\n'
            '}'
        )
        self.pushButton_2.setCheckable(True)
        self.pushButton_2.setAutoExclusive(False)

        self.verticalLayout.addWidget(self.pushButton_2)

        self.pushButton_3 = QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_3.setObjectName('pushButton_3')
        sizePolicy1.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy1)
        self.pushButton_3.setMinimumSize(QSize(60, 60))
        self.pushButton_3.setStyleSheet(
            'QPushButton{\n'
            '   background-color: rgb(70, 70, 70);\n'
            '   border-top-style: none;\n'
            '   border-bottom-style: solid;\n'
            '   border-bottom-color: black;\n'
            '   border-bottom-width: 1px;\n'
            '   border-left-style: none;\n'
            '   border-right-style: none;\n'
            '}\n'
            'QPushButton::checked{\n'
            '   background-color: rgb(255, 170, 0);\n'
            '}\n'
            'QPushButton:hover:!checked{\n'
            '   background-color: rgb(100, 100, 100);\n'
            '}'
        )
        self.pushButton_3.setCheckable(True)
        self.pushButton_3.setAutoExclusive(False)

        self.verticalLayout.addWidget(self.pushButton_3)

        self.pushButton_4 = QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_4.setObjectName('pushButton_4')
        sizePolicy1.setHeightForWidth(self.pushButton_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4.setSizePolicy(sizePolicy1)
        self.pushButton_4.setMinimumSize(QSize(60, 60))
        self.pushButton_4.setStyleSheet(
            'QPushButton{\n'
            '   background-color: rgb(70, 70, 70);\n'
            '   border: none;\n'
            '}\n'
            'QPushButton::checked{\n'
            '   background-color: rgb(255, 170, 0);\n'
            '}\n'
            'QPushButton:hover:!checked{\n'
            '   background-color: rgb(100, 100, 100);\n'
            '}'
        )
        self.pushButton_4.setCheckable(True)
        self.pushButton_4.setAutoExclusive(False)

        self.verticalLayout.addWidget(self.pushButton_4)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.panel_1 = QWidget(Form)
        self.panel_1.setObjectName('panel_1')
        self.panel_1.setGeometry(QRect(110, 50, 226, 141))
        self.panel_1.setStyleSheet('background-color: rgb(243, 243, 243);\n' 'border-radius: 4px;')
        self.gridLayout = QGridLayout(self.panel_1)
        self.gridLayout.setObjectName('gridLayout')
        self.panel_4 = QWidget(Form)
        self.panel_4.setObjectName('panel_4')
        self.panel_4.setGeometry(QRect(340, 200, 226, 141))
        self.panel_4.setStyleSheet('background-color: rgb(243, 243, 243);\n' 'border-radius: 4px;')
        self.gridLayout_4 = QGridLayout(self.panel_4)
        self.gridLayout_4.setObjectName('gridLayout_4')
        self.panel_2 = QWidget(Form)
        self.panel_2.setObjectName('panel_2')
        self.panel_2.setGeometry(QRect(340, 50, 226, 141))
        self.panel_2.setStyleSheet('background-color: rgb(243, 243, 243);\n' 'border-radius: 4px;')
        self.gridLayout_2 = QGridLayout(self.panel_2)
        self.gridLayout_2.setObjectName('gridLayout_2')
        self.panel_3 = QWidget(Form)
        self.panel_3.setObjectName('panel_3')
        self.panel_3.setGeometry(QRect(110, 200, 226, 141))
        self.panel_3.setStyleSheet('background-color: rgb(243, 243, 243);\n' 'border-radius: 4px;')
        self.gridLayout_3 = QGridLayout(self.panel_3)
        self.gridLayout_3.setObjectName('gridLayout_3')

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
        # setupUi

        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # anim
        self.toggleAnimation = QParallelAnimationGroup(self)
        scroll_anim = QPropertyAnimation(self.scrollArea, b'size')
        scroll_anim.setDuration(150)
        scroll_anim.setStartValue(QSize(60, 240))
        scroll_anim.setEndValue(QSize(0, 240))
        self.toggleAnimation.addAnimation(scroll_anim)

        button_anim = QPropertyAnimation(self.toolButton, b'geometry')
        button_anim.setDuration(150)
        button_anim.setStartValue(self.toolButton.geometry())
        button_anim.setEndValue(self.toolButton.geometry().adjusted(-60, 0, -60, 0))
        self.toggleAnimation.addAnimation(button_anim)

        self.toolButton.clicked.connect(self.clicktoolbutton)

        self.pushButton_1.clicked.connect(self.show_panel1)
        self.pushButton_2.clicked.connect(self.show_panel2)
        self.pushButton_3.clicked.connect(self.show_panel3)
        self.pushButton_4.clicked.connect(self.show_panel4)

        # init panel pos
        for button_id in range(1, 5):
            button = getattr(self, f'pushButton_{button_id}')
            panel_pos = QPoint(0 + 80, 60 * (button_id - 1))
            mapped = self.scrollArea.mapToParent(panel_pos)
            panel = getattr(self, f'panel_{button_id}')
            panel.move(mapped)
            panel.hide()

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate('Form', 'Form', None))
        self.toolButton.setText('')
        self.pushButton_1.setText(QCoreApplication.translate('Form', '1', None))
        self.pushButton_2.setText(QCoreApplication.translate('Form', '2', None))
        self.pushButton_3.setText(QCoreApplication.translate('Form', '3', None))
        self.pushButton_4.setText(QCoreApplication.translate('Form', '4', None))

    # retranslateUi

    def clicktoolbutton(self, checked):
        if checked:
            self.toolButton.setArrowType(Qt.RightArrow)
            self.toggleAnimation.setDirection(QAbstractAnimation.Forward)
        else:
            self.toolButton.setArrowType(Qt.LeftArrow)
            self.toggleAnimation.setDirection(QAbstractAnimation.Backward)
        self.toggleAnimation.start()

    def show_panel1(self):
        if self.pushButton_1.isChecked():
            for button_id in range(1, 5):
                if button_id != 1:
                    other_button = getattr(self, f'pushButton_{button_id}')
                    if other_button.isChecked():
                        other_button.click()
            self.panel_1.show()
        else:
            self.panel_1.hide()

    def show_panel2(self):
        if self.pushButton_2.isChecked():
            for button_id in range(1, 5):
                if button_id != 2:
                    other_button = getattr(self, f'pushButton_{button_id}')
                    if other_button.isChecked():
                        other_button.click()
            self.panel_2.show()
        else:
            self.panel_2.hide()

    def show_panel3(self):
        if self.pushButton_3.isChecked():
            for button_id in range(1, 5):
                if button_id != 3:
                    other_button = getattr(self, f'pushButton_{button_id}')
                    if other_button.isChecked():
                        other_button.click()
            self.panel_3.show()
        else:
            self.panel_3.hide()

    def show_panel4(self):
        if self.pushButton_4.isChecked():
            for button_id in range(1, 5):
                if button_id != 4:
                    other_button = getattr(self, f'pushButton_{button_id}')
                    if other_button.isChecked():
                        other_button.click()
            self.panel_4.show()
        else:
            self.panel_4.hide()
