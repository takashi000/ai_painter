# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialogdefaultprompts.ui'
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
    QGroupBox, QLabel, QPlainTextEdit, QSizePolicy,
    QWidget)

class Ui_DialogDefaultPrompts(object):
    def setupUi(self, DialogDefaultPrompts):
        if not DialogDefaultPrompts.objectName():
            DialogDefaultPrompts.setObjectName(u"DialogDefaultPrompts")
        DialogDefaultPrompts.resize(568, 315)
        self.groupBox = QGroupBox(DialogDefaultPrompts)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 10, 551, 271))
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 20, 49, 16))
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 140, 49, 16))
        self.plainTextEdit_defpos = QPlainTextEdit(self.groupBox)
        self.plainTextEdit_defpos.setObjectName(u"plainTextEdit_defpos")
        self.plainTextEdit_defpos.setGeometry(QRect(20, 40, 511, 91))
        self.plainTextEdit_defneg = QPlainTextEdit(self.groupBox)
        self.plainTextEdit_defneg.setObjectName(u"plainTextEdit_defneg")
        self.plainTextEdit_defneg.setGeometry(QRect(20, 160, 511, 91))
        self.buttonBox_defaultprompts = QDialogButtonBox(DialogDefaultPrompts)
        self.buttonBox_defaultprompts.setObjectName(u"buttonBox_defaultprompts")
        self.buttonBox_defaultprompts.setGeometry(QRect(400, 290, 164, 24))
        self.buttonBox_defaultprompts.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.retranslateUi(DialogDefaultPrompts)

        QMetaObject.connectSlotsByName(DialogDefaultPrompts)
    # setupUi

    def retranslateUi(self, DialogDefaultPrompts):
        DialogDefaultPrompts.setWindowTitle(QCoreApplication.translate("DialogDefaultPrompts", u"Default Prompts", None))
        self.groupBox.setTitle(QCoreApplication.translate("DialogDefaultPrompts", u"Default Prompts", None))
        self.label.setText(QCoreApplication.translate("DialogDefaultPrompts", u"Positive", None))
        self.label_2.setText(QCoreApplication.translate("DialogDefaultPrompts", u"Negative", None))
    # retranslateUi

