from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph.opengl import GLViewWidget


# overwrite slider
class VideoSlider(QSlider):
    def __init__(self, father):
        super().__init__(father)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            val = event.localPos().x()
            val = int(round(val / self.width() * self.maximum()))
            self.setValue(val)


class UiForm:
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName('Form')
        Form.resize(1920, 1000)
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.splitter = QSplitter(Form)
        self.splitter.setObjectName('splitter')
        self.splitter.setOrientation(Qt.Horizontal)
        self.scrollArea_l = QScrollArea(self.splitter)
        self.scrollArea_l.setObjectName('scrollArea_l')
        self.scrollArea_l.setWidgetResizable(True)
        self.scrollAreaWidgetContents_l = QWidget()
        self.scrollAreaWidgetContents_l.setObjectName('scrollAreaWidgetContents_l')
        self.scrollAreaWidgetContents_l.setGeometry(QRect(0, 0, 519, 980))
        self.verticalLayout_l = QVBoxLayout(self.scrollAreaWidgetContents_l)
        self.verticalLayout_l.setObjectName('verticalLayout_l')
        self.groupBox_l1 = QGroupBox(self.scrollAreaWidgetContents_l)
        self.groupBox_l1.setObjectName('groupBox_l1')
        font = QFont()
        font.setFamily('Arial')
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.font = font
        self.groupBox_l1.setFont(font)
        self.verticalLayout_l1 = QVBoxLayout(self.groupBox_l1)
        self.verticalLayout_l1.setObjectName('verticalLayout_l1')
        self.graphicsView_cam1 = GraphicsLayoutWidget(self.groupBox_l1)
        self.graphicsView_cam1.setObjectName('graphicsView_cam1')
        self.graphicsView_cam1.setMinimumSize(QSize(256, 256))

        self.verticalLayout_l1.addWidget(self.graphicsView_cam1)

        self.verticalLayout_l.addWidget(self.groupBox_l1)

        self.groupBox_l2 = QGroupBox(self.scrollAreaWidgetContents_l)
        self.groupBox_l2.setObjectName('groupBox_l2')
        self.groupBox_l2.setFont(font)
        self.verticalLayout_l2 = QVBoxLayout(self.groupBox_l2)
        self.verticalLayout_l2.setObjectName('verticalLayout_l2')
        self.graphicsView_cam3 = GraphicsLayoutWidget(self.groupBox_l2)
        self.graphicsView_cam3.setObjectName('graphicsView_cam3')
        self.graphicsView_cam3.setMinimumSize(QSize(256, 256))

        self.verticalLayout_l2.addWidget(self.graphicsView_cam3)

        self.verticalLayout_l.addWidget(self.groupBox_l2)

        self.scrollArea_l.setWidget(self.scrollAreaWidgetContents_l)
        self.splitter.addWidget(self.scrollArea_l)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName('layoutWidget')
        self.verticalLayout_9 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_9.setObjectName('verticalLayout_9')
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.splitter_show = QSplitter(self.layoutWidget)
        self.splitter_show.setObjectName('splitter_show')
        self.splitter_show.setOrientation(Qt.Vertical)
        self.openGLWidget = GLViewWidget(self.splitter_show)
        self.openGLWidget.setObjectName('openGLWidget')
        self.openGLWidget.setMinimumSize(QSize(480, 420))
        self.splitter_show.addWidget(self.openGLWidget)
        self.tabWidget = QTabWidget(self.splitter_show)
        self.tabWidget.setObjectName('tabWidget')
        self.tabWidget.setMinimumSize(QSize(480, 200))
        self.tab = QWidget()
        self.tab.setObjectName('tab')
        self.verticalLayout_7 = QVBoxLayout(self.tab)
        self.verticalLayout_7.setObjectName('verticalLayout_7')
        self.graphicsView_plot1 = GraphicsLayoutWidget(self.tab)
        self.graphicsView_plot1.setObjectName('graphicsView_plot1')

        self.verticalLayout_7.addWidget(self.graphicsView_plot1)

        self.tabWidget.addTab(self.tab, '')
        self.tab_2 = QWidget()
        self.tab_2.setObjectName('tab_2')
        self.verticalLayout_8 = QVBoxLayout(self.tab_2)
        self.verticalLayout_8.setObjectName('verticalLayout_8')
        self.graphicsView_plot2 = GraphicsLayoutWidget(self.tab_2)
        self.graphicsView_plot2.setObjectName('graphicsView_plot2')

        self.verticalLayout_8.addWidget(self.graphicsView_plot2)

        self.tabWidget.addTab(self.tab_2, '')
        self.splitter_show.addWidget(self.tabWidget)

        self.verticalLayout_9.addWidget(self.splitter_show)

        self.splitter_slider = QSplitter(self.layoutWidget)
        self.splitter_slider.setObjectName('splitter_slider')
        self.splitter_slider.setOrientation(Qt.Horizontal)
        self.spinBox_numFrame = QSpinBox(self.splitter_slider)
        self.spinBox_numFrame.setObjectName('spinBox_numFrame')
        self.splitter_slider.addWidget(self.spinBox_numFrame)
        self.label_numFrame = QLabel(self.splitter_slider)
        self.label_numFrame.setObjectName('label_numFrame')
        self.splitter_slider.addWidget(self.label_numFrame)
        self.horizontalSlider = VideoSlider(self.splitter_slider)
        self.horizontalSlider.setObjectName('horizontalSlider')
        self.horizontalSlider.setMinimumSize(QSize(240, 0))
        self.horizontalSlider.setOrientation(Qt.Horizontal)
        self.splitter_slider.addWidget(self.horizontalSlider)
        self.pushButton_back = QPushButton(self.splitter_slider)
        self.pushButton_back.setObjectName('pushButton_back')
        self.pushButton_back.setMinimumSize(QSize(20, 0))
        self.splitter_slider.addWidget(self.pushButton_back)
        self.pushButton_forward = QPushButton(self.splitter_slider)
        self.pushButton_forward.setObjectName('pushButton_forward')
        self.pushButton_forward.setMinimumSize(QSize(20, 0))
        self.splitter_slider.addWidget(self.pushButton_forward)

        self.pushButton_reset = QPushButton(self.splitter_slider)
        self.pushButton_reset.setObjectName('pushButton_reset')
        self.pushButton_reset.setMinimumSize(QSize(20, 0))
        self.splitter_slider.addWidget(self.pushButton_reset)

        self.pushButton_open = QPushButton(self.splitter_slider)
        self.pushButton_open.setObjectName('pushButton_open')
        self.splitter_slider.addWidget(self.pushButton_open)

        self.verticalLayout_9.addWidget(self.splitter_slider)

        self.splitter.addWidget(self.layoutWidget)
        self.scrollArea_r = QScrollArea(self.splitter)
        self.scrollArea_r.setObjectName('scrollArea_r')
        self.scrollArea_r.setWidgetResizable(True)
        self.scrollAreaWidgetContents_r = QWidget()
        self.scrollAreaWidgetContents_r.setObjectName('scrollAreaWidgetContents_r')
        self.scrollAreaWidgetContents_r.setGeometry(QRect(0, 0, 519, 980))
        self.verticalLayout_r = QVBoxLayout(self.scrollAreaWidgetContents_r)
        self.verticalLayout_r.setObjectName('verticalLayout_r')
        self.groupBox_r1 = QGroupBox(self.scrollAreaWidgetContents_r)
        self.groupBox_r1.setObjectName('groupBox_r1')
        self.groupBox_r1.setFont(font)
        self.verticalLayout_r1 = QVBoxLayout(self.groupBox_r1)
        self.verticalLayout_r1.setObjectName('verticalLayout_r1')
        self.graphicsView_cam2 = GraphicsLayoutWidget(self.groupBox_r1)
        self.graphicsView_cam2.setObjectName('graphicsView_cam2')
        self.graphicsView_cam2.setMinimumSize(QSize(256, 256))

        self.verticalLayout_r1.addWidget(self.graphicsView_cam2)

        self.verticalLayout_r.addWidget(self.groupBox_r1)

        self.groupBox_r2 = QGroupBox(self.scrollAreaWidgetContents_r)
        self.groupBox_r2.setObjectName('groupBox_r2')
        self.groupBox_r2.setFont(font)
        self.verticalLayout_r2 = QVBoxLayout(self.groupBox_r2)
        self.verticalLayout_r2.setObjectName('verticalLayout_r2')
        self.graphicsView_cam4 = GraphicsLayoutWidget(self.groupBox_r2)
        self.graphicsView_cam4.setObjectName('graphicsView_cam4')
        self.graphicsView_cam4.setMinimumSize(QSize(256, 256))

        self.verticalLayout_r2.addWidget(self.graphicsView_cam4)

        self.verticalLayout_r.addWidget(self.groupBox_r2)

        self.scrollArea_r.setWidget(self.scrollAreaWidgetContents_r)
        self.splitter.addWidget(self.scrollArea_r)

        self.horizontalLayout.addWidget(self.splitter)

        self.retranslateUi(Form)
        self.horizontalSlider.valueChanged.connect(self.spinBox_numFrame.setValue)
        self.spinBox_numFrame.valueChanged.connect(self.horizontalSlider.setValue)

        self.tabWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(Form)

    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate('Form', 'Form', None))
        self.groupBox_l1.setTitle(QCoreApplication.translate('Form', 'Camera 1', None))
        self.groupBox_l2.setTitle(QCoreApplication.translate('Form', 'Camera 3', None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab),
            QCoreApplication.translate('Form', 'Error', None),
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_2),
            QCoreApplication.translate('Form', 'Rec Ratio', None),
        )
        self.label_numFrame.setText(QCoreApplication.translate('Form', '/99', None))
        self.pushButton_back.setText(QCoreApplication.translate('Form', '<', None))
        self.pushButton_forward.setText(QCoreApplication.translate('Form', '>', None))
        self.pushButton_reset.setText(QCoreApplication.translate('Form', 'Reset', None))
        self.pushButton_open.setText(QCoreApplication.translate('Form', 'Open File', None))
        self.groupBox_r1.setTitle(QCoreApplication.translate('Form', 'Camera 2', None))
        self.groupBox_r2.setTitle(QCoreApplication.translate('Form', 'Camera 4', None))
        # retranslateUi

        self.groupBox_l1.setStyleSheet('QGroupBox:title {color: red;}')
        self.groupBox_r1.setStyleSheet('QGroupBox:title {color: green;}')
        self.groupBox_l2.setStyleSheet('QGroupBox:title {color: blue;}')
        self.groupBox_r2.setStyleSheet('QGroupBox:title {color: magenta;}')


class Curve_Menu_Form(QWidget):
    check_sig = pyqtSignal(list)

    def setupUi(self, Form, num_checkboxes):
        if not Form.objectName():
            Form.setObjectName('Form')
        # Form.resize(400, 300)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName('verticalLayout')

        assert num_checkboxes >= 1
        self.num_checkboxes = num_checkboxes

        for idx in range(num_checkboxes):
            name = f'checkBox_h{idx}'
            checkbox = QCheckBox(Form)
            checkbox.setObjectName(name)
            self.verticalLayout.addWidget(checkbox)
            setattr(self, name, checkbox)

        def outer(checkbox_id):
            def chb(state):
                self.check_sig.emit([checkbox_id, state])

            return chb

        for idx in range(self.num_checkboxes):
            getattr(self, f'checkBox_h{idx}').stateChanged.connect(outer(idx))
            getattr(self, f'checkBox_h{idx}').setChecked(True)

        self.retranslateUi(Form)
        QMetaObject.connectSlotsByName(Form)

        self.id = 1

    def setID(self, id):
        """
        Set instance id in case there are many instances, so id is also emitted.
        """
        self.id = id

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate('Form', 'Form', None))
        for idx in range(self.num_checkboxes):
            getattr(self, f'checkBox_h{idx}').setText(
                QCoreApplication.translate('Form', f'h{idx}', None)
            )

    # retranslateUi
    # def setCheckedGroup(self, num):
    #     for idx in range(num):
    #         getattr(self, f'checkBox_h{idx}').setChecked(True)
