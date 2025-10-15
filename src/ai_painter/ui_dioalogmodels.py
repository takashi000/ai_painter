# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialogmodels.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLayout, QPlainTextEdit, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget)

class Ui_DialogModels(object):
    def setupUi(self, DialogModels):
        if not DialogModels.objectName():
            DialogModels.setObjectName(u"DialogModels")
        DialogModels.resize(421, 868)
        DialogModels.setMinimumSize(QSize(421, 0))
        DialogModels.setMaximumSize(QSize(421, 16777215))
        self.buttonBox_dialogmodels = QDialogButtonBox(DialogModels)
        self.buttonBox_dialogmodels.setObjectName(u"buttonBox_dialogmodels")
        self.buttonBox_dialogmodels.setGeometry(QRect(250, 840, 164, 24))
        self.buttonBox_dialogmodels.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)
        self.groupBox = QGroupBox(DialogModels)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 10, 401, 201))
        self.formLayoutWidget = QWidget(self.groupBox)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(10, 20, 381, 171))
        self.formLayout = QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.formLayout.setLabelAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.formLayoutWidget)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.plainTextEdit_checkpoint = QPlainTextEdit(self.formLayoutWidget)
        self.plainTextEdit_checkpoint.setObjectName(u"plainTextEdit_checkpoint")
        self.plainTextEdit_checkpoint.setMinimumSize(QSize(0, 25))
        self.plainTextEdit_checkpoint.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_2.addWidget(self.plainTextEdit_checkpoint)

        self.toolButton_checkpoint = QToolButton(self.formLayoutWidget)
        self.toolButton_checkpoint.setObjectName(u"toolButton_checkpoint")

        self.horizontalLayout_2.addWidget(self.toolButton_checkpoint)


        self.formLayout.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_2)

        self.label_2 = QLabel(self.formLayoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_2)

        self.label_3 = QLabel(self.formLayoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_3)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.plainTextEdit_vae = QPlainTextEdit(self.formLayoutWidget)
        self.plainTextEdit_vae.setObjectName(u"plainTextEdit_vae")
        self.plainTextEdit_vae.setMinimumSize(QSize(0, 25))
        self.plainTextEdit_vae.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_3.addWidget(self.plainTextEdit_vae)

        self.toolButton_vae = QToolButton(self.formLayoutWidget)
        self.toolButton_vae.setObjectName(u"toolButton_vae")

        self.horizontalLayout_3.addWidget(self.toolButton_vae)


        self.formLayout.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_3)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.plainTextEdit_lora = QPlainTextEdit(self.formLayoutWidget)
        self.plainTextEdit_lora.setObjectName(u"plainTextEdit_lora")
        self.plainTextEdit_lora.setMinimumSize(QSize(0, 0))
        self.plainTextEdit_lora.setMaximumSize(QSize(16777215, 25))
        self.plainTextEdit_lora.setInputMethodHints(Qt.InputMethodHint.ImhMultiLine)

        self.horizontalLayout.addWidget(self.plainTextEdit_lora)

        self.toolButton_lora = QToolButton(self.formLayoutWidget)
        self.toolButton_lora.setObjectName(u"toolButton_lora")

        self.horizontalLayout.addWidget(self.toolButton_lora)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_5 = QLabel(self.formLayoutWidget)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_6.addWidget(self.label_5)

        self.plainTextEdit_loraname = QPlainTextEdit(self.formLayoutWidget)
        self.plainTextEdit_loraname.setObjectName(u"plainTextEdit_loraname")
        self.plainTextEdit_loraname.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_6.addWidget(self.plainTextEdit_loraname)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_4 = QLabel(self.formLayoutWidget)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_4.addWidget(self.label_4)

        self.doubleSpinBox_loraweght = QDoubleSpinBox(self.formLayoutWidget)
        self.doubleSpinBox_loraweght.setObjectName(u"doubleSpinBox_loraweght")
        self.doubleSpinBox_loraweght.setMaximum(1.000000000000000)
        self.doubleSpinBox_loraweght.setSingleStep(0.010000000000000)
        self.doubleSpinBox_loraweght.setValue(1.000000000000000)

        self.horizontalLayout_4.addWidget(self.doubleSpinBox_loraweght)


        self.verticalLayout.addLayout(self.horizontalLayout_4)


        self.formLayout.setLayout(2, QFormLayout.ItemRole.FieldRole, self.verticalLayout)

        self.groupBox_2 = QGroupBox(DialogModels)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 220, 401, 531))
        self.horizontalLayoutWidget_4 = QWidget(self.groupBox_2)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(10, 20, 381, 501))
        self.horizontalLayout_7 = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.formLayout_3.setLabelAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.label_20 = QLabel(self.horizontalLayoutWidget_4)
        self.label_20.setObjectName(u"label_20")

        self.formLayout_3.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_20)

        self.label_21 = QLabel(self.horizontalLayoutWidget_4)
        self.label_21.setObjectName(u"label_21")

        self.formLayout_3.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_21)

        self.label_22 = QLabel(self.horizontalLayoutWidget_4)
        self.label_22.setObjectName(u"label_22")

        self.formLayout_3.setWidget(13, QFormLayout.ItemRole.LabelRole, self.label_22)

        self.label_23 = QLabel(self.horizontalLayoutWidget_4)
        self.label_23.setObjectName(u"label_23")

        self.formLayout_3.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_23)

        self.label_24 = QLabel(self.horizontalLayoutWidget_4)
        self.label_24.setObjectName(u"label_24")

        self.formLayout_3.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_24)

        self.label_25 = QLabel(self.horizontalLayoutWidget_4)
        self.label_25.setObjectName(u"label_25")

        self.formLayout_3.setWidget(11, QFormLayout.ItemRole.LabelRole, self.label_25)

        self.label_26 = QLabel(self.horizontalLayoutWidget_4)
        self.label_26.setObjectName(u"label_26")

        self.formLayout_3.setWidget(12, QFormLayout.ItemRole.LabelRole, self.label_26)

        self.label_27 = QLabel(self.horizontalLayoutWidget_4)
        self.label_27.setObjectName(u"label_27")

        self.formLayout_3.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_27)

        self.label_28 = QLabel(self.horizontalLayoutWidget_4)
        self.label_28.setObjectName(u"label_28")

        self.formLayout_3.setWidget(10, QFormLayout.ItemRole.LabelRole, self.label_28)

        self.label_29 = QLabel(self.horizontalLayoutWidget_4)
        self.label_29.setObjectName(u"label_29")

        self.formLayout_3.setWidget(8, QFormLayout.ItemRole.LabelRole, self.label_29)

        self.label_30 = QLabel(self.horizontalLayoutWidget_4)
        self.label_30.setObjectName(u"label_30")

        self.formLayout_3.setWidget(5, QFormLayout.ItemRole.LabelRole, self.label_30)

        self.label_31 = QLabel(self.horizontalLayoutWidget_4)
        self.label_31.setObjectName(u"label_31")

        self.formLayout_3.setWidget(6, QFormLayout.ItemRole.LabelRole, self.label_31)

        self.label_32 = QLabel(self.horizontalLayoutWidget_4)
        self.label_32.setObjectName(u"label_32")

        self.formLayout_3.setWidget(7, QFormLayout.ItemRole.LabelRole, self.label_32)

        self.label_33 = QLabel(self.horizontalLayoutWidget_4)
        self.label_33.setObjectName(u"label_33")

        self.formLayout_3.setWidget(9, QFormLayout.ItemRole.LabelRole, self.label_33)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.plainTextEdit_canny = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_canny.setObjectName(u"plainTextEdit_canny")
        self.plainTextEdit_canny.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_8.addWidget(self.plainTextEdit_canny)

        self.toolButton_canny = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_canny.setObjectName(u"toolButton_canny")

        self.horizontalLayout_8.addWidget(self.toolButton_canny)


        self.formLayout_3.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.plainTextEdit_ip2p = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_ip2p.setObjectName(u"plainTextEdit_ip2p")
        self.plainTextEdit_ip2p.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_9.addWidget(self.plainTextEdit_ip2p)

        self.toolButton_ip2p = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_ip2p.setObjectName(u"toolButton_ip2p")

        self.horizontalLayout_9.addWidget(self.toolButton_ip2p)


        self.formLayout_3.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.plainTextEdit_inpaint = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_inpaint.setObjectName(u"plainTextEdit_inpaint")
        self.plainTextEdit_inpaint.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_10.addWidget(self.plainTextEdit_inpaint)

        self.toolButton_inpaint = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_inpaint.setObjectName(u"toolButton_inpaint")

        self.horizontalLayout_10.addWidget(self.toolButton_inpaint)


        self.formLayout_3.setLayout(2, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.plainTextEdit_mlsd = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_mlsd.setObjectName(u"plainTextEdit_mlsd")
        self.plainTextEdit_mlsd.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_11.addWidget(self.plainTextEdit_mlsd)

        self.toolButton_mlsd = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_mlsd.setObjectName(u"toolButton_mlsd")

        self.horizontalLayout_11.addWidget(self.toolButton_mlsd)


        self.formLayout_3.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.plainTextEdit_depth = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_depth.setObjectName(u"plainTextEdit_depth")
        self.plainTextEdit_depth.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_12.addWidget(self.plainTextEdit_depth)

        self.toolButton_depth = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_depth.setObjectName(u"toolButton_depth")

        self.horizontalLayout_12.addWidget(self.toolButton_depth)


        self.formLayout_3.setLayout(4, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.plainTextEdit_normalbae = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_normalbae.setObjectName(u"plainTextEdit_normalbae")
        self.plainTextEdit_normalbae.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_13.addWidget(self.plainTextEdit_normalbae)

        self.toolButton_normalbae = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_normalbae.setObjectName(u"toolButton_normalbae")

        self.horizontalLayout_13.addWidget(self.toolButton_normalbae)


        self.formLayout_3.setLayout(5, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_13)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.plainTextEdit_seg = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_seg.setObjectName(u"plainTextEdit_seg")
        self.plainTextEdit_seg.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_14.addWidget(self.plainTextEdit_seg)

        self.toolButton_seg = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_seg.setObjectName(u"toolButton_seg")

        self.horizontalLayout_14.addWidget(self.toolButton_seg)


        self.formLayout_3.setLayout(6, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_14)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.plainTextEdit_lineart = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_lineart.setObjectName(u"plainTextEdit_lineart")
        self.plainTextEdit_lineart.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_15.addWidget(self.plainTextEdit_lineart)

        self.toolButton_lineart = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_lineart.setObjectName(u"toolButton_lineart")

        self.horizontalLayout_15.addWidget(self.toolButton_lineart)


        self.formLayout_3.setLayout(7, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_15)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.plainTextEdit_lineartanime = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_lineartanime.setObjectName(u"plainTextEdit_lineartanime")
        self.plainTextEdit_lineartanime.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_16.addWidget(self.plainTextEdit_lineartanime)

        self.toolButton_lineartanime = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_lineartanime.setObjectName(u"toolButton_lineartanime")

        self.horizontalLayout_16.addWidget(self.toolButton_lineartanime)


        self.formLayout_3.setLayout(8, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_16)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.plainTextEdit_openpose = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_openpose.setObjectName(u"plainTextEdit_openpose")
        self.plainTextEdit_openpose.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_17.addWidget(self.plainTextEdit_openpose)

        self.toolButton_openpose = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_openpose.setObjectName(u"toolButton_openpose")

        self.horizontalLayout_17.addWidget(self.toolButton_openpose)


        self.formLayout_3.setLayout(9, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_17)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.plainTextEdit_scribble = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_scribble.setObjectName(u"plainTextEdit_scribble")
        self.plainTextEdit_scribble.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_18.addWidget(self.plainTextEdit_scribble)

        self.toolButton_scribble = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_scribble.setObjectName(u"toolButton_scribble")

        self.horizontalLayout_18.addWidget(self.toolButton_scribble)


        self.formLayout_3.setLayout(10, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_18)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.plainTextEdit_softedge = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_softedge.setObjectName(u"plainTextEdit_softedge")
        self.plainTextEdit_softedge.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_19.addWidget(self.plainTextEdit_softedge)

        self.toolButton_softedge = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_softedge.setObjectName(u"toolButton_softedge")

        self.horizontalLayout_19.addWidget(self.toolButton_softedge)


        self.formLayout_3.setLayout(11, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_19)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.plainTextEdit_shuffle = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_shuffle.setObjectName(u"plainTextEdit_shuffle")
        self.plainTextEdit_shuffle.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_20.addWidget(self.plainTextEdit_shuffle)

        self.toolButton_shuffle = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_shuffle.setObjectName(u"toolButton_shuffle")

        self.horizontalLayout_20.addWidget(self.toolButton_shuffle)


        self.formLayout_3.setLayout(12, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_20)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.plainTextEdit_tile = QPlainTextEdit(self.horizontalLayoutWidget_4)
        self.plainTextEdit_tile.setObjectName(u"plainTextEdit_tile")
        self.plainTextEdit_tile.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_21.addWidget(self.plainTextEdit_tile)

        self.toolButton_tile = QToolButton(self.horizontalLayoutWidget_4)
        self.toolButton_tile.setObjectName(u"toolButton_tile")

        self.horizontalLayout_21.addWidget(self.toolButton_tile)


        self.formLayout_3.setLayout(13, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_21)


        self.horizontalLayout_7.addLayout(self.formLayout_3)

        self.groupBox_3 = QGroupBox(DialogModels)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(10, 750, 401, 81))
        self.widget = QWidget(self.groupBox_3)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(10, 30, 379, 31))
        self.formLayout_4 = QFormLayout(self.widget)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.formLayout_4.setLabelAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)
        self.formLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label_34 = QLabel(self.widget)
        self.label_34.setObjectName(u"label_34")

        self.formLayout_4.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_34)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.plainTextEdit_upernetseg = QPlainTextEdit(self.widget)
        self.plainTextEdit_upernetseg.setObjectName(u"plainTextEdit_upernetseg")
        self.plainTextEdit_upernetseg.setMaximumSize(QSize(16777215, 25))

        self.horizontalLayout_23.addWidget(self.plainTextEdit_upernetseg)

        self.toolButton_upernetseg = QToolButton(self.widget)
        self.toolButton_upernetseg.setObjectName(u"toolButton_upernetseg")

        self.horizontalLayout_23.addWidget(self.toolButton_upernetseg)


        self.formLayout_4.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_23)


        self.retranslateUi(DialogModels)

        QMetaObject.connectSlotsByName(DialogModels)
    # setupUi

    def retranslateUi(self, DialogModels):
        DialogModels.setWindowTitle(QCoreApplication.translate("DialogModels", u"Models", None))
        self.groupBox.setTitle(QCoreApplication.translate("DialogModels", u"Checkpoint", None))
        self.label.setText(QCoreApplication.translate("DialogModels", u"Checkpoint", None))
        self.toolButton_checkpoint.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.label_2.setText(QCoreApplication.translate("DialogModels", u"VAE", None))
        self.label_3.setText(QCoreApplication.translate("DialogModels", u"LoRa", None))
        self.toolButton_vae.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_lora.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.label_5.setText(QCoreApplication.translate("DialogModels", u"Name", None))
        self.label_4.setText(QCoreApplication.translate("DialogModels", u"Weight", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("DialogModels", u"ControlNet", None))
        self.label_20.setText(QCoreApplication.translate("DialogModels", u"canny", None))
        self.label_21.setText(QCoreApplication.translate("DialogModels", u"ip2p", None))
        self.label_22.setText(QCoreApplication.translate("DialogModels", u"tile", None))
        self.label_23.setText(QCoreApplication.translate("DialogModels", u"inpaint", None))
        self.label_24.setText(QCoreApplication.translate("DialogModels", u"mlsd", None))
        self.label_25.setText(QCoreApplication.translate("DialogModels", u"softedge", None))
        self.label_26.setText(QCoreApplication.translate("DialogModels", u"shuffle", None))
        self.label_27.setText(QCoreApplication.translate("DialogModels", u"depth", None))
        self.label_28.setText(QCoreApplication.translate("DialogModels", u"scribble", None))
        self.label_29.setText(QCoreApplication.translate("DialogModels", u"lineart_anime", None))
        self.label_30.setText(QCoreApplication.translate("DialogModels", u"normalbae", None))
        self.label_31.setText(QCoreApplication.translate("DialogModels", u"seg", None))
        self.label_32.setText(QCoreApplication.translate("DialogModels", u"lineart", None))
        self.label_33.setText(QCoreApplication.translate("DialogModels", u"openpose", None))
        self.toolButton_canny.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_ip2p.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_inpaint.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_mlsd.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_depth.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_normalbae.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_seg.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_lineart.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_lineartanime.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_openpose.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_scribble.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_softedge.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_shuffle.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.toolButton_tile.setText(QCoreApplication.translate("DialogModels", u"...", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("DialogModels", u"UperNet", None))
        self.label_34.setText(QCoreApplication.translate("DialogModels", u"seg", None))
        self.toolButton_upernetseg.setText(QCoreApplication.translate("DialogModels", u"...", None))
    # retranslateUi

