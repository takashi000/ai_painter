from __future__ import annotations
from PySide6.QtWidgets import (
    QApplication, 
    QMainWindow,
    QMenuBar,
    QMenu,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QToolButton,
    QDialogButtonBox,
    QToolBar,
    QSlider,
    QCheckBox,
    QTextEdit,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QStackedWidget,
    QGraphicsView, 
    QGraphicsScene, 
    QGraphicsPixmapItem, 
    QWidget, 
    QDialog,
    QFileDialog, 
    QStyle, 
    QColorDialog,
    QDialog,
)
from PySide6.QtCore import (
    Qt,
    Slot,
    QStandardPaths,
    QBuffer,
    QIODevice,
    QThread,
    Signal,
    QPoint,
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QMouseEvent,
    QPaintEvent,
    QPen,
    QAction,
    QPainter,
    QColor,
    QIcon,
    QKeySequence,
)
from PIL import Image
import numpy as np
import sys
import io
from .ui_paint import Ui_MainWindow
from .ui_dioalogmodels import Ui_DialogModels
from .ui_dialogdefaultprompts import Ui_DialogDefaultPrompts
from .ui_about import Ui_DialogAbout
from .pic import DiffusersGenerate
from .paintsystem import PaintSystem


class ImageGen(QThread):
    run_finished = Signal()
    def __init__(self, win):
       super(ImageGen, self).__init__()
       self.win = win
    def run(self):
        positive_prompt = self.win.posEdit.toPlainText()
        negative_prompt = self.win.negEdit.toPlainText()
        guidance_scale = self.win.dspinBoxGuidance.value()
        check_seed = self.win.checkBoxSeed.isChecked()
        steps = self.win.spinBoxStep.value()
        seed = self.win.spinBoxSeed.value() if check_seed else -1
        image, seed_ret = self.win.imagegen.generate_image(
            prompt=positive_prompt, 
            neg_prompt=negative_prompt,
            guidance=guidance_scale,
            steps=steps,
            seed=seed,
        )
        self.win.spinBoxSeed.setValue(seed_ret)
        self.win.generated_image = self.win.pilimage_to_qimage(image)
        self.run_finished.emit()

class ImageGenControlNet(QThread):
    run_finished = Signal()
    def __init__(self, win):
       super(ImageGenControlNet, self).__init__()
       self.win = win
    def run(self):
        positive_prompt = self.win.posEdit.toPlainText()
        negative_prompt = self.win.negEdit.toPlainText()
        cnet_type = self.win.cnet_checkpoint[1]
        guidance_scale = self.win.dspinBoxGuidance.value()
        steps = self.win.spinBoxStep.value()
        eta = self.win.dspinBoxEta.value()
        check_seed = self.win.checkBoxSeed.isChecked()
        check_blur = self.win.checkBoxBlur.isChecked()
        check_quant = self.win.checkBoxQuant.isChecked()
        check_bayer = self.win.checkBoxBayer.isChecked()
        seed = self.win.spinBoxSeed.value() if check_seed else -1
        th_low = self.win.spinBoxThLow.value()
        th_high = self.win.spinBoxThHigh.value()
        num_colors = self.win.spinBoxQuant.value()
        ksize = self.win.spinBoxBlur.value()
        image_in = self.win.base_image
        image_mask = self.win.mask_image if cnet_type == "inpaint" else None
        openpose_alias = self.win.comboBoxAlias.currentText()
        radius = self.win.spinBoxRadius.value()
        thickness = self.win.spinBoxThickness.value()
        kpt_thr = self.win.dspinBoxKptthr.value()
        shuffle_k = self.win.spinBoxShufflek.value()
        block = self.win.checkBoxBlock.isChecked()
        block_k = self.win.spinBoxBlockk.value()

        image, seed_ret = self.win.imagegen.generate_image_controlnet(
            prompt=positive_prompt, 
            image_in=image_in,
            neg_prompt=negative_prompt,
            guidance=guidance_scale,
            steps=steps,
            eta=eta,
            seed=seed,
            strength=1.0,
            cnet_type=cnet_type,
            low_threshold=th_low,
            high_threshold=th_high,
            blur=check_blur,
            ksize=ksize,
            quant=check_quant,
            bayer=check_bayer,
            num_colors=num_colors,
            image_mask=image_mask,
            openpose_alias=openpose_alias, 
            radius=radius, 
            thickness=thickness, 
            kpt_thr=kpt_thr, 
            shuffle_k=shuffle_k, 
            block=block, 
            block_k=block_k
        )
        self.win.spinBoxSeed.setValue(seed_ret)
        self.win.generated_image = self.win.pilimage_to_qimage(image)
        self.run_finished.emit()

class RealoadControlNet(QThread):
    run_finished = Signal()
    def __init__(self, win):
       super(RealoadControlNet, self).__init__()
       self.win = win
    def run(self):
        checkpoint = self.win.cnet_checkpoint
        self.win.imagegen.reload_controlnet_model(checkpoint[0], checkpoint[1])
        self.run_finished.emit()

class ReloadCheckpoint(QThread):
    run_finished = Signal()
    def __init__(self, win):
        super(ReloadCheckpoint, self).__init__()
        self.win = win
    def run(self):
        checkpoint = self.win.cnet_checkpoint
        self.win.imagegen.reload_model(
            model=self.win.system.systemconf["Checkpoint"]["checkpoint"],
            lora=self.win.system.systemconf["Checkpoint"]["lora"]["file"],
            loradir=self.win.system.systemconf["Checkpoint"]["lora"]["lora_dir"],
            lora_name=self.win.system.systemconf["Checkpoint"]["lora"]["lora_name"],
            lora_weight=self.win.system.systemconf["Checkpoint"]["lora"]["lora_weight"],
            vae=self.win.system.systemconf["Checkpoint"]["vae"],
        )
        self.win.imagegen.reload_controlnet_model(checkpoint[0], checkpoint[1])
        self.run_finished.emit()

class ImagePreviewDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canny Image Preview")

        self.layout = QVBoxLayout()

        self.image_label = QLabel("No Image")
        self.image_label.setFixedSize(512, 512)
        self.layout.addWidget(self.image_label)

        self.setLayout(self.layout)

    def show_image(self, pixmap: QPixmap):
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.IgnoreAspectRatio))
        self.image_label.setText("")

class ViewWidget(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(530, 530)
        self.scene = QGraphicsScene(self)
        self.pixmap_item = None
        self.pixmap = QPixmap(512, 512)
        self.qimage = self.pixmap.toImage()

        self.scene.addItem(QGraphicsPixmapItem(self.pixmap))
        self.setScene(self.scene)

    def set_image(self, image: QImage):
        if image.isNull():
            self.pixmap = QPixmap(512, 512)
        else:
            self.pixmap = QPixmap.fromImage(image)

        self.qimage = self.pixmap.toImage()

        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.pixmap_item)
        self.setScene(self.scene)
        
    def save(self, filename: str):
        """ save pixmap to filename """
        self.pixmap.save(filename)

class PainterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(512, 512)
        self.pixmap = QPixmap(512, 512)
        self.pixmap.fill(Qt.GlobalColor.white)

        self.pixmap_org = self.pixmap
        self.pixmap_mask = QPixmap(512, 512)
        self.pixmap_mask.fill(Qt.GlobalColor.black)

        self.pixmap_pen = QPixmap(512, 512)
        self.pixmap_pen.fill(Qt.GlobalColor.white)

        self.pixmap_history = self.history()

        self.previous_pos = None
        self.current_pos = None

        self.pen = QPen()
        self.pen.setWidth(10)
        self.pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        
        self.eraser = QPen()
        self.eraser.setColor(Qt.GlobalColor.white)
        self.eraser.setWidth(10)
        self.eraser.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.eraser.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        self.painter = QPainter()
       
        self.draw_modes = ["free", "line", "circle", "ellipse", "square", "erase"]
        self.draw_mode = "free"
        self.fill_modes = ["none", "base", "draw"]
        self.fill_mode = "none"
        self.fill_on = False

    def paintEvent(self, event: QPaintEvent):
        """Override method from QWidget

        Paint the Pixmap into the widget

        """
        with QPainter(self) as painter:
            painter.drawPixmap(0, 0, self.pixmap)
            match self.draw_mode:
                case "free":
                    if self.fill_on:
                        # 塗りつぶし
                        self.update()
                case "line":
                    if self.previous_pos and self.current_pos:
                        painter.drawLine(self.previous_pos, self.current_pos)
                    self.update()
                case "square":
                    if self.previous_pos and self.current_pos:
                        if self.previous_pos.y() <= self.current_pos.y():
                            painter.drawRect(self.previous_pos.x(), self.previous_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                        else:
                            painter.drawRect(self.current_pos.x(), self.current_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                    self.update()
                case "ellipse":
                    if self.previous_pos and self.current_pos:
                        if self.previous_pos.y() <= self.current_pos.y():
                            painter.drawEllipse(self.previous_pos.x(), self.previous_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                        else:
                            painter.drawEllipse(self.current_pos.x(), self.current_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                    self.update()
                case "circle":
                    if self.previous_pos and self.current_pos:
                        r_x =  abs(self.previous_pos.x()-self.current_pos.x()) // 2
                        r_y =  abs(self.previous_pos.y()-self.current_pos.y()) // 2
                        r = int((r_x**2 + r_y**2)**0.5)
                        painter.drawEllipse(self.previous_pos.x() - r, self.previous_pos.y() - r, r*2, r*2)
                    self.update()
    def mousePressEvent(self, event: QMouseEvent):
        """Override from QWidget

        Called when user clicks on the mouse

        """
        if event.button() == Qt.MouseButton.LeftButton:            
            self.previous_pos = event.position().toPoint()
            self.current_pos = self.previous_pos
            QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Override method from QWidget

        Called when user moves and clicks on the mouse

        """
        if event.buttons() & Qt.MouseButton.LeftButton:
            match self.draw_mode:
                case "free":
                    self.current_pos = event.position().toPoint()
                    if not self.fill_on:
                        for pixmap in [self.pixmap, self.pixmap_pen]:
                            self.painter.begin(pixmap)
                            self.painter.setPen(self.pen)
                            self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                            self.painter.drawLine(self.previous_pos, self.current_pos)
                            self.painter.end()

                        # mask draw
                        prevcolor = self.pen.color()
                        self.pen.setColor(Qt.GlobalColor.white)
                        self.painter.begin(self.pixmap_mask)
                        self.painter.setPen(self.pen)
                        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                        self.painter.drawLine(self.previous_pos, self.current_pos)
                        self.painter.end()
                        self.pen.setColor(prevcolor)

                    self.previous_pos = self.current_pos
                    self.update()
                case "line":
                    self.current_pos = event.position().toPoint()
                case "square":
                    self.current_pos = event.position().toPoint()
                case "ellipse":
                    self.current_pos = event.position().toPoint()
                case "circle":
                    self.current_pos = event.position().toPoint()
                case "erase":
                    self.current_pos = event.position().toPoint()
                    for pixmap in [self.pixmap, self.pixmap_pen]:
                        self.painter.begin(pixmap)
                        self.painter.setPen(self.eraser)
                        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                        self.painter.drawLine(self.previous_pos, self.current_pos)
                        self.painter.end()

                    # mask draw
                    prevcolor = self.pen.color()
                    self.pen.setColor(Qt.GlobalColor.black)
                    self.painter.begin(self.pixmap_mask)
                    self.painter.setPen(self.pen)
                    self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                    self.painter.drawLine(self.previous_pos, self.current_pos)
                    self.painter.end()
                    self.pen.setColor(prevcolor)

                    self.previous_pos = self.current_pos
                    self.update()
            
            QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Override method from QWidget

        Called when user releases the mouse

        """
        if event.button() == Qt.MouseButton.LeftButton:
            match self.draw_mode:
                case "free":
                    if self.fill_on:
                        # 塗りつぶし
                        self.pixmap, self.pixmap_mask, self.pixmap_pen = self.flood_fill(self.pixmap, self.pixmap_mask, self.pixmap_pen, QPoint(self.current_pos), self.pen.color(), self.fill_mode)
                case "line":
                    for pixmap in [self.pixmap, self.pixmap_pen]:
                        self.painter.begin(pixmap)
                        self.painter.setPen(self.pen)
                        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                        self.painter.drawLine(self.previous_pos, self.current_pos)
                        self.painter.end()
                    
                    # mask draw
                    prevcolor = self.pen.color()
                    self.pen.setColor(Qt.GlobalColor.white)
                    self.painter.begin(self.pixmap_mask)
                    self.painter.setPen(self.pen)
                    self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                    self.painter.drawLine(self.previous_pos, self.current_pos)
                    self.painter.end()
                    self.pen.setColor(prevcolor)

                case "square":
                    for pixmap in [self.pixmap, self.pixmap_pen]:
                        self.painter.begin(pixmap)
                        self.painter.setPen(self.pen)
                        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                        if self.previous_pos.y() <= self.current_pos.y():
                            self.painter.drawRect(self.previous_pos.x(), self.previous_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                        else:
                            self.painter.drawRect(self.current_pos.x(), self.current_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                        self.painter.end()
                    
                    # mask draw
                    prevcolor = self.pen.color()
                    self.pen.setColor(Qt.GlobalColor.white)
                    self.painter.begin(self.pixmap_mask)
                    self.painter.setPen(self.pen)
                    self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                    if self.previous_pos.y() <= self.current_pos.y():
                        self.painter.drawRect(self.previous_pos.x(), self.previous_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                    else:
                        self.painter.drawRect(self.current_pos.x(), self.current_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                    self.painter.end()
                    self.pen.setColor(prevcolor)
                
                case "ellipse":
                    for pixmap in [self.pixmap, self.pixmap_pen]:
                        self.painter.begin(pixmap)
                        self.painter.setPen(self.pen)
                        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                        if self.previous_pos.y() <= self.current_pos.y():
                            self.painter.drawEllipse(self.previous_pos.x(), self.previous_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                        else:
                            self.painter.drawEllipse(self.current_pos.x(), self.current_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                        self.painter.end()
                    
                    # mask draw
                    prevcolor = self.pen.color()
                    self.pen.setColor(Qt.GlobalColor.white)
                    self.painter.begin(self.pixmap_mask)
                    self.painter.setPen(self.pen)
                    self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                    if self.previous_pos.y() <= self.current_pos.y():
                        self.painter.drawEllipse(self.previous_pos.x(), self.previous_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                    else:
                        self.painter.drawEllipse(self.current_pos.x(), self.current_pos.y(), abs(self.previous_pos.x()-self.current_pos.x()), abs(self.previous_pos.y()-self.current_pos.y()))
                    self.painter.end()
                    self.pen.setColor(prevcolor)

                case "circle":
                    for pixmap in [self.pixmap, self.pixmap_pen]:
                        self.painter.begin(pixmap)
                        self.painter.setPen(self.pen)
                        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                        r_x =  abs(self.previous_pos.x()-self.current_pos.x()) // 2
                        r_y =  abs(self.previous_pos.y()-self.current_pos.y()) // 2
                        r = int((r_x**2 + r_y**2)**0.5)
                        self.painter.drawEllipse(self.previous_pos.x() - r, self.previous_pos.y() - r, r*2, r*2)
                        self.painter.end()
                    
                    # mask draw
                    prevcolor = self.pen.color()
                    self.pen.setColor(Qt.GlobalColor.white)
                    self.painter.begin(self.pixmap_mask)
                    self.painter.setPen(self.pen)
                    self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
                    r_x =  abs(self.previous_pos.x()-self.current_pos.x()) // 2
                    r_y =  abs(self.previous_pos.y()-self.current_pos.y()) // 2
                    r = int((r_x**2 + r_y**2)**0.5)
                    self.painter.drawEllipse(self.previous_pos.x() - r, self.previous_pos.y() - r, r*2, r*2)
                    self.painter.end()
                    self.pen.setColor(prevcolor)              

        self.previous_pos = None
        self.current_pos = None
        self.pixmap_history = self.history(histories=self.pixmap_history, action="push")
        QWidget.mouseReleaseEvent(self, event)

    def save(self, filename: str):
        """ save pixmap to filename """
        self.pixmap.save(filename)

    def load(self, filename: str):
        """ load pixmap from filename """
        self.pixmap.load(filename)
        self.pixmap = self.pixmap.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
        self.pixmap_org.load(filename)
        self.pixmap_org = self.pixmap.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
        self.pixmap_mask.fill(Qt.GlobalColor.black)
        self.pixmap_pen.fill(Qt.GlobalColor.white)
        self.pixmap_history = self.history(self.pixmap_history, action="push")
        self.update()

    def load_from_qimage(self, image: QImage):
        """ load pixmap from QImage """
        self.pixmap = QPixmap.fromImage(image)
        self.pixmap = self.pixmap.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
        self.pixmap_org = QPixmap.fromImage(image)
        self.pixmap_org = self.pixmap.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
        self.pixmap_mask.fill(Qt.GlobalColor.black)
        self.pixmap_pen.fill(Qt.GlobalColor.white)
        self.pixmap_history = self.history(self.pixmap_history, action="push")
        self.update()

    def clear(self):
        """ Clear the pixmap """
        self.pixmap.fill(Qt.GlobalColor.white)
        self.pixmap_org.fill(Qt.GlobalColor.white)
        self.pixmap_mask.fill(Qt.GlobalColor.black)
        self.pixmap_pen.fill(Qt.GlobalColor.white)
        self.pixmap_history = self.history(self.pixmap_history, action="push")
        self.update()

    def undo(self):
        """ Undo the pixmap """
        self.pixmap_history = self.history(self.pixmap_history, action="undo")
        (self.pixmap, self.pixmap_mask, self.pixmap_pen) = self.pixmap_history["images"]
        self.update()

    def redo(self):
        """ Redo the pixmap """
        self.pixmap_history = self.history(self.pixmap_history, action="redo")
        (self.pixmap, self.pixmap_mask, self.pixmap_pen) = self.pixmap_history["images"]
        self.update()

    def penbold(self, value:int):
        self.pen.setWidth(value)
        self.eraser.setWidth(value)
        self.update()

    def setfill(self, on:bool, mode:str="none"):
        self.fill_mode = mode if mode in self.fill_modes else "none"
        self.fill_on = on if self.fill_mode != "none" else False
        self.update()

    def drawmode(self, mode:str="free"):
        self.draw_mode = mode if mode in self.draw_modes else "free"
        self.update()

    def flood_fill(self, pixmap:QPixmap, pixmap_mask:QPixmap, pixmap_pen:QPixmap, start_point:QPoint, new_color:QColor, mode:str="draw") -> (QPixmap, QPixmap, QPixmap):
        # drawモードのときはペンで描画した範囲を塗りつぶす
        # baseモードのときは読み込んだ画像の範囲を塗りつぶす
        # QPixmapをNumPy配列に変換
        image = pixmap.toImage()
        image_mask = pixmap_mask.toImage()
        image_pen = pixmap_pen.toImage()

        width, height = image.width(), image.height()

        # 開始点の色を取得
        start_color = image_pen.pixelColor(start_point) if mode == "draw" else image.pixelColor(start_point)

        # NumPyで配列化
        data = np.array(image.bits()).reshape((height, width, 4))
        data_mask = np.array(image_mask.bits()).reshape((height, width, 4))
        data_pen = np.array(image_pen.bits()).reshape((height, width, 4))

        # 塗りつぶしの条件
        if (start_color.blue(), start_color.green(), start_color.red()) == (new_color.blue(), new_color.green(), new_color.red()):
            return pixmap, pixmap_mask, pixmap_pen  # 同じ色の場合は何もしない

        stack = [(start_point.x(), start_point.y())]
        
        while stack:
            x, y = stack.pop()
            
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            current_color = data_pen[y, x][:3] if mode == "draw" else data[y, x][:3]

            # 現在の色が開始色の場合のみ、塗りつぶす
            if tuple(current_color) == (start_color.blue(), start_color.green(), start_color.red()):
                data[y, x][:3] = (new_color.blue(), new_color.green(), new_color.red())  # 塗りつぶし
                data_mask[y, x][:3] = (255, 255, 255)  # マスク画像塗りつぶし
                data_pen[y, x][:3] = (new_color.blue(), new_color.green(), new_color.red())  # ペン領域の塗りつぶし

                # 隣接ピクセルをスタックに追加
                stack.append((x + 1, y))
                stack.append((x - 1, y))
                stack.append((x, y + 1))
                stack.append((x, y - 1))

        # NumPy配列を戻してQPixmapに
        new_image = QPixmap.fromImage(image)
        new_image_mask = QPixmap.fromImage(image_mask)
        new_image_pen = QPixmap.fromImage(image_pen)
         # 透明に初期化
        new_image.fill(QColor(0, 0, 0, 0))
        new_image_mask.fill(QColor(0, 0, 0, 0))
        new_image.fill(QColor(0, 0, 0, 0))
        new_image.convertFromImage(QImage(data.data, width, height, QImage.Format_ARGB32))
        new_image_mask.convertFromImage(QImage(data_mask.data, width, height, QImage.Format_ARGB32))
        new_image_pen.convertFromImage(QImage(data_pen.data, width, height, QImage.Format_ARGB32))

        return new_image, new_image_mask, new_image_pen
    
    def history(self, histories:dict={}, action:str="create", index:int=-1, max:int=10):
        # action: create, push, pop, undo, redo, delete
        # index: -1 最新
        # max: 履歴の最大保持数 最大保持数を超えた場合は一番古い履歴を削除
        # 戻り値：　push, delte -> list pop -> tuple

        history_image = (None, None, None)
        history_images = histories
        
        if histories:
            if len(history_images["history"]) > max:
                elems = history_images["history"][0]
                history_images["history"] = history_images["history"][1:]
                for elem in elems:
                    del elem
                del elems

        match action:
            case "create":
                history_image = (self.pixmap.copy(), self.pixmap_mask.copy(), self.pixmap_pen.copy())
                history_images =  history_images = {
                    "history":[history_image],
                    "images":history_image,
                    "data": {
                        "latest": 0,
                        "current": 0,
                    },
                }
            case "push":
                history_image = (self.pixmap.copy(), self.pixmap_mask.copy(), self.pixmap_pen.copy())
                if index >= 0 :
                    history_images["history"].insert(index, history_image)
                    for elem in history_images["history"][index + 1:]:
                        del elem
                    del history_images["history"][index + 1:]
                else:
                    index_current = history_images["data"]["current"]
                    if index_current < history_images["data"]["latest"]:
                        history_images["history"].insert(index_current + 1, history_image)
                        for elem in history_images["history"][index_current + 2:]:
                            del elem
                        del history_images["history"][index_current + 2:]
                    else:
                        history_images["history"].append(history_image)
                latest = len(history_images["history"]) - 1
                history_images["data"]["latest"] = latest
                history_images["data"]["current"] = latest
                history_images["images"] = history_images["history"][latest]
            case "pop":
                latest = len(history_images["history"]) - 1
                if -1 <= index and latest >= index:
                    history_images["data"]["latest"] = latest
                    history_images["data"]["current"] = index
                    history_images["images"] = history_images["history"][index]
            case "undo":
                index_prev = history_images["data"]["current"] - 1
                if index_prev >= 0:
                    (pixmap, pixmap_mask, pixmap_pen) = history_images["history"][index_prev]
                    history_images["images"] = (pixmap.copy(), pixmap_mask.copy(), pixmap_pen.copy())
                    history_images["data"]["current"] = index_prev
            case "redo":
                index_next = history_images["data"]["current"] + 1
                latest = history_images["data"]["latest"]
                if index_next <= latest:
                    (pixmap, pixmap_mask, pixmap_pen) = history_images["history"][index_next]
                    history_images["images"] = (pixmap.copy(), pixmap_mask.copy(), pixmap_pen.copy())
                    history_images["data"]["current"] = index_next
            case "delete":
                latest = len(history_images["history"]) - 1
                if -1 <= index and latest >= index:
                    for elem in history_images["history"][index]:
                        del elem
                    del history_images["history"][index]
                current = history_images["data"]["current"]
                if current > latest:
                    history_images["data"]["current"] = latest - 1
                history_images["images"] = history_images["history"][latest - 1]
            case _:
                pass
            
        return history_images

class DialogModels(QDialog):
    def __init__(self, parent=None):
        super(DialogModels, self).__init__(parent)
        self.ui = Ui_DialogModels()
        self.ui.setupUi(self)

        self.parent = parent

        self.plainTextEditCheckpoint: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_checkpoint")
        self.plainTextEditVAE: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_vae")
        self.plainTextEditLora: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_lora")
        self.plainTextEditLoraName: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_loraname")
        self.plainTextEditCanny: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_canny")
        self.plainTextEditIp2p: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_ip2p")
        self.plainTextEditInpaint: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_inpaint")
        self.plainTextEditMlsd: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_mlsd")
        self.plainTextEditDepth: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_depth")
        self.plainTextEditNormalBae: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_normalbae")
        self.plainTextEditSeg: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_seg")
        self.plainTextEditLineart: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_lineart")
        self.plainTextEditLineartAnime: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_lineartanime")
        self.plainTextEditOpenpose: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_openpose")
        self.plainTextEditScribble: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_scribble")
        self.plainTextEditSoftEdge: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_softedge")
        self.plainTextEditShuffle: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_shuffle")
        self.plainTextEditTile: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_tile")
        self.plainTextEditUperNetSeg: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_upernetseg")
        self.toolButtonCheckpoint: QToolButton = self.findChild(QToolButton, "toolButton_checkpoint")
        self.toolButtonVAE: QToolButton = self.findChild(QToolButton, "toolButton_vae")
        self.toolButtonLora: QToolButton = self.findChild(QToolButton, "toolButton_lora")
        self.toolButtonCanny: QToolButton = self.findChild(QToolButton, "toolButton_canny")
        self.toolButtonIp2p: QToolButton = self.findChild(QToolButton, "toolButton_ip2p")
        self.toolButtonInpaint: QToolButton = self.findChild(QToolButton, "toolButton_inpaint")
        self.toolButtonMlsd: QToolButton = self.findChild(QToolButton, "toolButton_mlsd")
        self.toolButtonDepth: QToolButton = self.findChild(QToolButton, "toolButton_depth")
        self.toolButtonNormalbae: QToolButton = self.findChild(QToolButton, "toolButton_normalbae")
        self.toolButtonSeg: QToolButton = self.findChild(QToolButton, "toolButton_seg")
        self.toolButtonLineart: QToolButton = self.findChild(QToolButton, "toolButton_lineart")
        self.toolButtonLineartAnime: QToolButton = self.findChild(QToolButton, "toolButton_lineartanime")
        self.toolButtonOpenpose: QToolButton = self.findChild(QToolButton, "toolButton_openpose")
        self.toolButtonScribble: QToolButton = self.findChild(QToolButton, "toolButton_scribble")
        self.toolButtonSoftEdge: QToolButton = self.findChild(QToolButton, "toolButton_softedge")
        self.toolButtonShuffle: QToolButton = self.findChild(QToolButton, "toolButton_shuffle")
        self.toolButtonTile: QToolButton = self.findChild(QToolButton, "toolButton_tile")
        self.toolButtonUperNetSeg: QToolButton = self.findChild(QToolButton, "toolButton_upernetseg")
        self.dSpinBoxLoRaWeght: QDoubleSpinBox = self.findChild(QDoubleSpinBox, "doubleSpinBox_loraweght")
        self.buttonBoxDialogModels: QDialogButtonBox = self.findChild(QDialogButtonBox, "buttonBox_dialogmodels")

        self.toolButtonCheckpoint.clicked.connect(self.on_open)
        self.toolButtonVAE.clicked.connect(self.on_open)
        self.toolButtonLora.clicked.connect(self.on_open)
        self.toolButtonCanny.clicked.connect(self.on_open)
        self.toolButtonIp2p.clicked.connect(self.on_open)
        self.toolButtonInpaint.clicked.connect(self.on_open)
        self.toolButtonMlsd.clicked.connect(self.on_open)
        self.toolButtonDepth.clicked.connect(self.on_open)
        self.toolButtonNormalbae.clicked.connect(self.on_open)
        self.toolButtonSeg.clicked.connect(self.on_open)
        self.toolButtonLineart.clicked.connect(self.on_open)
        self.toolButtonLineartAnime.clicked.connect(self.on_open)
        self.toolButtonOpenpose.clicked.connect(self.on_open)
        self.toolButtonScribble.clicked.connect(self.on_open)
        self.toolButtonSoftEdge.clicked.connect(self.on_open)
        self.toolButtonShuffle.clicked.connect(self.on_open)
        self.toolButtonTile.clicked.connect(self.on_open)
        self.toolButtonUperNetSeg.clicked.connect(self.on_open)
        self.buttonBoxDialogModels.accepted.connect(self.on_accepted)
        self.buttonBoxDialogModels.rejected.connect(self.reject)
        
        self.settings = self.parent.system.load_systemconf()

        # Checkpointの設定
        checkpoint = self.settings['Checkpoint']
        self.plainTextEditCheckpoint.setPlainText(checkpoint['checkpoint'])
        self.plainTextEditVAE.setPlainText(checkpoint['vae'])

        # Loraの設定
        lora = checkpoint['lora']
        self.plainTextEditLora.setPlainText(lora['file'])
        self.plainTextEditLoraName.setPlainText(lora['lora_name'])
        self.dSpinBoxLoRaWeght.setValue(lora['lora_weight'])

        # ControlNetの設定
        controlnet = self.settings['ControlNet']
        self.plainTextEditCanny.setPlainText(controlnet['canny'])
        self.plainTextEditIp2p.setPlainText(controlnet['ip2p'])
        self.plainTextEditInpaint.setPlainText(controlnet['inpaint'])
        self.plainTextEditMlsd.setPlainText(controlnet['mlsd'])
        self.plainTextEditDepth.setPlainText(controlnet['depth'])
        self.plainTextEditNormalBae.setPlainText(controlnet['normalbae'])
        self.plainTextEditSeg.setPlainText(controlnet['seg'])
        self.plainTextEditLineart.setPlainText(controlnet['lineart'])
        self.plainTextEditLineartAnime.setPlainText(controlnet['lineart_anime'])
        self.plainTextEditOpenpose.setPlainText(controlnet['openpose'])
        self.plainTextEditScribble.setPlainText(controlnet['scribble'])
        self.plainTextEditSoftEdge.setPlainText(controlnet['softedge'])
        self.plainTextEditShuffle.setPlainText(controlnet['shuffle'])
        self.plainTextEditTile.setPlainText(controlnet['tile'])
        
        # UperNetの設定
        upernet = self.settings['UperNet']
        self.plainTextEditUperNetSeg.setPlainText(upernet['seg'])


    def on_open(self):

        sender = self.sender()
        objname = sender.objectName()
        confpath = str(self.parent.system.confpath["Models"])

        dialog = QFileDialog(self, "Open file")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setDefaultSuffix("safetensors")
        dialog.setDirectory(confpath)
        if objname == "toolButton_checkpoint" or objname == "toolButton_vae" or objname == "toolButton_lora":
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        else:
            dialog.setFileMode(QFileDialog.FileMode.Directory)

        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            if dialog.selectedFiles():
                filepath = dialog.selectedFiles()[0]
                match objname:
                    case "toolButton_checkpoint":
                        self.plainTextEditCheckpoint.setPlainText(filepath)
                        self.plainTextEditCheckpoint.show()
                    case "toolButton_vae":
                        self.plainTextEditVAE.setPlainText(filepath)
                        self.plainTextEditVAE.show()
                    case "toolButton_lora":
                        self.plainTextEditLora.setPlainText(filepath)
                        self.plainTextEditLora.show()
                    case "toolButton_canny":
                        self.plainTextEditCanny.setPlainText(filepath)
                        self.plainTextEditCanny.show()
                    case "toolButton_ip2p":
                        self.plainTextEditIp2p.setPlainText(filepath)
                        self.plainTextEditIp2p.show()
                    case "toolButton_inpaint":
                        self.plainTextEditInpaint.setPlainText(filepath)
                        self.plainTextEditInpaint.show()
                    case "toolButton_mlsd":
                        self.plainTextEditMlsd.setPlainText(filepath)
                        self.plainTextEditMlsd.show()
                    case "toolButton_depth":
                        self.plainTextEditDepth.setPlainText(filepath)
                        self.plainTextEditDepth.show()
                    case "toolButton_normalbae":
                        self.plainTextEditNormalBae.setPlainText(filepath)
                        self.plainTextEditNormalBae.show()
                    case "toolButton_seg":
                        self.plainTextEditSeg.setPlainText(filepath)
                        self.plainTextEditSeg.show()
                    case "toolButton_lineart":
                        self.plainTextlineart.setPlainText(filepath)
                        self.plainTextEditLineart.show()
                    case "toolButton_lineartanime":
                        self.plainTextEditLineartAnime.setPlainText(filepath)
                        self.plainTextEditLineartAnime.show()
                    case "toolButton_openpose":
                        self.plainTextEditOpenpose.setPlainText(filepath)
                        self.plainTextEditOpenpose.show()
                    case "toolButton_scribble":
                        self.plainTextEditScribble.setPlainText(filepath)
                        self.plainTextEditScribble.show()
                    case "toolButton_softedge":
                        self.plainTextEditSoftEdge.setPlainText(filepath)
                        self.plainTextEditSoftEdge.show()
                    case "toolButton_shuffle":
                        self.plainTextEditShuffle.setPlainText(filepath)
                        self.plainTextEditShuffle.show()
                    case "toolButton_tile":
                        self.plainTextEditTile.setPlainText(filepath)
                        self.plainTextEditTile.show()
                    case "toolButton_upernetseg":
                        self.plainTextEditUperNetSeg.setPlainText(filepath)
                        self.plainTextEditUperNetSeg.show()
    
    def on_accepted(self):

        conf_checkpoint = self.parent.system.conf_system_checkpoint()
        conf_controlnet = self.parent.system.conf_system_controlnet()
        conf_upernet = self.parent.system.conf_system_upernet()

        conf_checkpoint["checkpoint"] = self.plainTextEditCheckpoint.toPlainText()
        conf_checkpoint["vae"] = self.plainTextEditVAE.toPlainText()
        conf_checkpoint["lora"]["file"] = self.plainTextEditLora.toPlainText()
        conf_checkpoint["lora"]["lora_dir"] = str(self.parent.system.modelspath["Lora"])
        conf_checkpoint["lora"]["lora_name"] = self.plainTextEditLoraName.toPlainText()
        conf_checkpoint["lora"]["lora_weight"] = self.dSpinBoxLoRaWeght.value()

        conf_controlnet["canny"] = self.plainTextEditCanny.toPlainText()
        conf_controlnet["ip2p"] = self.plainTextEditIp2p.toPlainText()
        conf_controlnet["inpaint"] = self.plainTextEditInpaint.toPlainText()
        conf_controlnet["mlsd"] = self.plainTextEditMlsd.toPlainText()
        conf_controlnet["depth"] = self.plainTextEditDepth.toPlainText()
        conf_controlnet["normalbae"] = self.plainTextEditNormalBae.toPlainText()
        conf_controlnet["seg"] = self.plainTextEditSeg.toPlainText()
        conf_controlnet["lineart"] = self.plainTextEditLineart.toPlainText()
        conf_controlnet["lineart_anime"] = self.plainTextEditLineartAnime.toPlainText()
        conf_controlnet["openpose"] = self.plainTextEditOpenpose.toPlainText()
        conf_controlnet["scribble"] = self.plainTextEditScribble.toPlainText()
        conf_controlnet["softedge"] = self.plainTextEditSoftEdge.toPlainText()
        conf_controlnet["shuffle"] = self.plainTextEditShuffle.toPlainText()
        conf_controlnet["tile"] = self.plainTextEditTile.toPlainText()

        conf_upernet["seg"] = self.plainTextEditUperNetSeg.toPlainText()

        self.parent.system.conf_system_checkpoint(conf_checkpoint)
        self.parent.system.conf_system_controlnet(conf_controlnet)
        self.parent.system.conf_system_upernet(conf_upernet)
        self.parent.system.save_systemconf()

        self.accept()

class DialogDefaultPrompts(QDialog):
    def __init__(self, parent=None):
        super(DialogDefaultPrompts, self).__init__(parent)
        self.ui = Ui_DialogDefaultPrompts()
        self.ui.setupUi(self)

        self.parent = parent

        self.plainTextEditDefPos: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_defpos")
        self.plainTextEditDefNeg: QPlainTextEdit = self.findChild(QPlainTextEdit, "plainTextEdit_defneg")
        self.buttonBoxDialogDefPrompts: QDialogButtonBox = self.findChild(QDialogButtonBox, "buttonBox_defaultprompts")

        self.buttonBoxDialogDefPrompts.accepted.connect(self.on_accepted)
        self.buttonBoxDialogDefPrompts.rejected.connect(self.reject)

        self.settings = self.parent.system.load_systemconf()

        # Default Promptsの設定
        defaultprompts = self.settings["DefaultPrompts"]
        self.plainTextEditDefPos.setPlainText(defaultprompts["Positive"])
        self.plainTextEditDefNeg.setPlainText(defaultprompts["Negative"])

    def on_accepted(self):
        
        conf_defaultprompts = self.parent.system.conf_system_defaultprompts()

        conf_defaultprompts["Positive"] = self.plainTextEditDefPos.toPlainText()
        conf_defaultprompts["Negative"] = self.plainTextEditDefNeg.toPlainText()

        self.parent.system.conf_system_defaultprompts(conf_defaultprompts)
        self.parent.system.save_systemconf()

        self.accept()

class DialogAbout(QDialog):
    def __init__(self, parent=None):
        super(DialogAbout, self).__init__(parent)
        self.ui = Ui_DialogAbout()
        self.ui.setupUi(self)

        self.parent = parent

        self.pushButtonAboutOK: QPushButton = self.findChild(QPushButton, "pushButton_aboutok")

        self.pushButtonAboutOK.clicked.connect(self.accept)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("AIPainter")

        self.system = PaintSystem()
        self.systemconf = self.system()

        self.menubar: QMenuBar = self.findChild(QMenuBar, "menubar")
        self.menufile: QMenu = self.findChild(QMenu, "menufile")
        self.menuedit: QMenu = self.findChild(QMenu, "menuedit")
        self.menugenerate: QMenu = self.findChild(QMenu, "menugenerate")
        self.menuoptions: QMenu = self.findChild(QMenu, "menuoptions")
        self.menuhelp: QMenu = self.findChild(QMenu, "menuhelp")
        self.horizontal_layout: QHBoxLayout = self.findChild(QHBoxLayout, "horizontalLayout")
        self.verticalLayoutColor: QVBoxLayout = self.findChild(QVBoxLayout, "verticalLayout_color")
        self.graphicsview: QGraphicsView = self.findChild(QGraphicsView, "graphicsview")
        self.posEdit: QTextEdit = self.findChild(QTextEdit, "posEdit")
        self.negEdit: QTextEdit = self.findChild(QTextEdit, "negEdit")
        self.spinBoxSeed: QSpinBox = self.findChild(QSpinBox, "spinBox_seed")
        self.spinBoxStep: QSpinBox = self.findChild(QSpinBox, "spinBox_step")
        self.spinBoxThLow: QSpinBox = self.findChild(QSpinBox, "spinBox_thlow")
        self.spinBoxThHigh: QSpinBox = self.findChild(QSpinBox, "spinBox_thhigh")
        self.spinBoxBlur: QSpinBox = self.findChild(QSpinBox, "spinBox_Blur")
        self.spinBoxQuant: QSpinBox = self.findChild(QSpinBox, "spinBox_Quant")
        self.spinBoxRadius: QSpinBox = self.findChild(QSpinBox, "spinBox_radius")
        self.spinBoxThickness: QSpinBox = self.findChild(QSpinBox, "spinBox_thickness")
        self.spinBoxShufflek: QSpinBox = self.findChild(QSpinBox, "spinBox_shufflek")
        self.spinBoxBlockk: QSpinBox = self.findChild(QSpinBox, "spinBox_blockk")
        self.dspinBoxGuidance: QDoubleSpinBox = self.findChild(QDoubleSpinBox, "doubleSpinBox_guidance")
        self.dspinBoxEta: QDoubleSpinBox = self.findChild(QDoubleSpinBox, "doubleSpinBox_eta")
        self.dspinBoxKptthr: QDoubleSpinBox = self.findChild(QDoubleSpinBox, "doubleSpinBox_kptthr")
        self.dspinBoxStrength: QDoubleSpinBox = self.findChild(QDoubleSpinBox, "doubleSpinBox_strength")
        self.checkBoxSeed: QCheckBox = self.findChild(QCheckBox, "checkBox_seed")
        self.checkControlNet: QCheckBox = self.findChild(QCheckBox, "checkBox_cnet")
        self.checkBoxBlur: QCheckBox = self.findChild(QCheckBox, "checkBox_Blur")
        self.checkBoxQuant: QCheckBox = self.findChild(QCheckBox, "checkBox_Quant")
        self.checkBoxBayer: QCheckBox = self.findChild(QCheckBox, "checkBox_Bayer")
        self.checkBoxFill: QCheckBox = self.findChild(QCheckBox, "checkBox_fillenable")
        self.checkBoxBlock: QCheckBox = self.findChild(QCheckBox, "checkBox_Block")
        self.comboBoxCnetcp: QComboBox = self.findChild(QComboBox, "comboBox_cnetcp")
        self.comboBoxAlias: QComboBox = self.findChild(QComboBox, "comboBox_alias")
        self.sliderPenBold: QSlider = self.findChild(QSlider, "horizontalSlider_penbold")
        self.previewButton: QPushButton = self.findChild(QPushButton, "pushButton_previeweffect")
        self.colorButton: QPushButton = self.findChild(QPushButton, "pushButton_color")
        self.labelstatus: QLabel = self.findChild(QLabel, "label_staus")
        self.radioButtonFree: QRadioButton = self.findChild(QRadioButton, "radioButton_free")
        self.radioButtonCircle: QRadioButton = self.findChild(QRadioButton, "radioButton_circle")
        self.radioButtonEllipse: QRadioButton = self.findChild(QRadioButton, "radioButton_ellipse")
        self.radioButtonLine: QRadioButton = self.findChild(QRadioButton, "radioButton_line")
        self.radioButtonSquare: QRadioButton = self.findChild(QRadioButton, "radioButton_square")
        self.radioButtonErase: QRadioButton = self.findChild(QRadioButton, "radioButton_erase")
        self.radioButtonFillDraw: QRadioButton = self.findChild(QRadioButton, "radioButton_filldraw")
        self.radioButtonFillBase: QRadioButton = self.findChild(QRadioButton, "radioButton_fillbase")
        self.stackedWidgetCnet: QStackedWidget = self.findChild(QStackedWidget, "stackedWidget_cnet")

        self.painter_widget = PainterWidget()
        self.view_widget = ViewWidget()

        self.generated_image:QImage = None
        self.base_image:Image = self.qimage_to_pil(self.view_widget.qimage)
        self.mask_image:Image = None
        
        self.menufile.addAction(
            qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),  # noqa: F821
            "Open", self.on_open
        )
        self.menufile.addAction(
             qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),  # noqa: F821
            "Save", self.on_save
        )
        self.menufile.addAction(
            qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),  # noqa: F821
            "Save Generated File", self.on_save_generated
        )
        self.menuedit.addAction(
            qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),  # noqa: F821
            "Load from generated image",
            self.on_load_from_generated_image,
        )
        self.menuedit.addAction(
             qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton),  # noqa: F821
             "Undo", self.painter_widget.undo,
        )
        self.menuedit.addAction(
             qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton),  # noqa: F821
             "Redo", self.painter_widget.redo,
        )
        self.menuedit.addAction(
            qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton),  # noqa: F821
            "Clear",
            self.painter_widget.clear,
        )
        self.menugenerate.addAction(
            qApp.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload),  # noqa: F821
            "Generate Image",
            self.on_generate_image,
        )
        self.menuoptions.addAction(
            qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),  # noqa: F821
            "Models",
            self.on_models
        )
        self.menuoptions.addAction(
            qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),  # noqa: F821
            "Default Prompts",
            self.on_defaultprompts
        )
        self.menuhelp.addAction(
             qApp.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),  # noqa: F821
             "About",
             self.on_about
        )

        #self.menubar.addSeparator()

        self.bar = QToolBar("")
        self.bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.verticalLayoutColor.addWidget(self.bar)
        self.color_action = QAction("Color",self)
        self.color_action.setObjectName("color_clicked")
        self.color_action.triggered.connect(self.on_color_clicked)
        self.bar.addAction(self.color_action)

        self.horizontal_layout.addWidget(self.view_widget)
        self.horizontal_layout.addWidget(self.painter_widget)
        
        self.color = Qt.GlobalColor.black
        self.set_color(self.color)

        self.mime_type_filters = ["image/png", "image/jpeg"]

        self.comboBoxCnetcp.currentTextChanged.connect(self.on_cnetcp_text_changed)
        self.sliderPenBold.valueChanged.connect(self.on_pen_bold_changed)
        self.previewButton.clicked.connect(self.on_previeweffect)
        self.radioButtonFree.clicked.connect(self.on_drawmodefree)
        self.radioButtonCircle.clicked.connect(self.on_drawmodecircle)
        self.radioButtonEllipse.clicked.connect(self.on_drawmodeellipse)
        self.radioButtonLine.clicked.connect(self.on_drawmodeline)
        self.radioButtonSquare.clicked.connect(self.on_drawmodesquare)
        self.radioButtonErase.clicked.connect(self.on_drawmodeerase)
        self.radioButtonFillDraw.clicked.connect(self.on_fillmode)
        self.radioButtonFillBase.clicked.connect(self.on_fillmode)
        self.checkBoxFill.clicked.connect(self.on_fillmode)

        self.positive_prompt = self.system.systemconf["DefaultPrompts"]["Positive"]
        self.negative_prompt = self.system.systemconf["DefaultPrompts"]["Negative"]
        self.guidancescale = 7.5
        self.steps = 20
        self.eta = 1.0
        self.seed = -1
        self.strength = 1.0

        self.cnet_checkpoints = self.load_cnet_checkpoints()
        self.cnet_checkpoint = self.cnet_checkpoints[0]

        self.cnet_stackwidgets = [
            ("canny", 1),
            ("ip2p", 0),
            ("inpaint", 2),
            ("mlsd", 0),
            ("depth", 0),
            ("normalbae", 0),
            ("seg", 0),
            ("lineart", 1),
            ("lineart_anime", 1),
            ("openpose", 3),
            ("scribble", 1),
            ("softedge", 1),
            ("shuffle", 4),
            ("tile", 5)
        ]
        self.cnet_stackwidget = self.cnet_stackwidgets[0]

        self.imagegen = DiffusersGenerate(
            model=self.system.systemconf["Checkpoint"]["checkpoint"],
            lora=self.system.systemconf["Checkpoint"]["lora"]["file"],
            loradir=self.system.systemconf["Checkpoint"]["lora"]["lora_dir"],
            lora_name=self.system.systemconf["Checkpoint"]["lora"]["lora_name"],
            lora_weight=self.system.systemconf["Checkpoint"]["lora"]["lora_weight"],
            vae=self.system.systemconf["Checkpoint"]["vae"],
            model_segmentation=self.system.systemconf["UperNet"]["seg"],
            model_cnet=self.cnet_checkpoint[0])
        self.running = False
        self.loading = False
        self.imagegen_run = ImageGen(self)
        self.imagegen_run.run_finished.connect(self.generate_complete)
        self.imagegen_controlnet_run = ImageGenControlNet(self)
        self.imagegen_controlnet_run.run_finished.connect(self.generate_complete)

        self.reload_checkpoint_run = ReloadCheckpoint(self)
        self.reload_checkpoint_run.run_finished.connect(self.reload_complete)
        self.reload_controlnet_run = RealoadControlNet(self)
        self.reload_controlnet_run.run_finished.connect(self.reload_complete)

        self.posEdit.setPlainText(self.positive_prompt)
        self.negEdit.setPlainText(self.negative_prompt)

        self.spinBoxSeed.setValue(self.seed)
        self.spinBoxStep.setValue(self.steps)
        self.dspinBoxEta.setValue(self.eta)
        self.dspinBoxGuidance.setValue(self.guidancescale)

        self.spinBoxThLow.setValue(100)
        self.spinBoxThHigh.setValue(200)
        self.spinBoxQuant.setValue(2)
        self.spinBoxBlur.setValue(3)
        self.spinBoxRadius.setValue(4)
        self.spinBoxThickness.setValue(1)
        self.spinBoxShufflek.setValue(5)
        self.spinBoxBlockk.setValue(6)
        self.dspinBoxKptthr.setValue(0.5)
        self.dspinBoxStrength.setValue(1.0)
        self.sliderPenBold.setValue(10)

    @Slot()
    def on_save(self):

        dialog = QFileDialog(self, "Save File")
        dialog.setMimeTypeFilters(self.mime_type_filters)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("png")
        dialog.setDirectory(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        )

        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            if dialog.selectedFiles():
                self.painter_widget.save(dialog.selectedFiles()[0])
    @Slot()
    def on_open(self):

        dialog = QFileDialog(self, "Save File")
        dialog.setMimeTypeFilters(self.mime_type_filters)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setDefaultSuffix("png")
        dialog.setDirectory(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        )

        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            if dialog.selectedFiles():
                self.painter_widget.load(dialog.selectedFiles()[0])

    def on_save_generated(self):

        dialog = QFileDialog(self, "Save Generated File")
        dialog.setMimeTypeFilters(self.mime_type_filters)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setDefaultSuffix("png")
        dialog.setDirectory(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        )

        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            if dialog.selectedFiles():
                self.view_widget.save(dialog.selectedFiles()[0])

    def on_generate_image(self):
        if self.running != True and self.loading != True:
            self.running = True
            if self.checkControlNet.isChecked():
                if self.cnet_checkpoint[1] == "inpaint":
                    self.base_image = self.qimage_to_pil(self.painter_widget.pixmap_org.toImage())
                    self.mask_image = self.qimage_to_pil(self.painter_widget.pixmap_mask.toImage())
                else:
                    self.base_image = self.qimage_to_pil(self.painter_widget.pixmap.toImage())
                self.imagegen_controlnet_run.start()
            else:
                self.imagegen_run.start()
        self.labelstatus.setText("Generating Image...")

    def generate_complete(self):
        self.view_widget.set_image(self.generated_image)
        self.running = False
        self.labelstatus.setText("")

    def on_load_from_generated_image(self):
        if self.generated_image is not None:
            self.painter_widget.load_from_qimage(self.generated_image)
    
    def on_models(self):
        dialog = DialogModels(self)

        if dialog.exec() == DialogModels.DialogCode.Accepted:
            self.loadding = True
            self.labelstatus.setText("Loading Model...")
            self.cnet_checkpoints = self.load_cnet_checkpoints()
            checkpoint = [cp for cp in self.cnet_checkpoints if cp[1] == self.comboBoxCnetcp.currentText()][0]
            self.cnet_checkpoint = checkpoint if checkpoint else self.cnet_checkpoints[0]
            self.reload_checkpoint_run.start()
        else:
            pass

    def on_defaultprompts(self):
        dialog = DialogDefaultPrompts(self)

        if dialog.exec() == DialogModels.DialogCode.Accepted:
            pass
        else:
            pass

    def on_about(self):
        dialog = DialogAbout(self)

        if dialog.exec() == DialogAbout.DialogCode.Accepted:
            pass
        else:
            pass

    def on_color_clicked(self):

        color = QColorDialog.getColor(self.color, self)

        if color:
            self.set_color(color)

    def on_pen_bold_changed(self, value:int):
        self.painter_widget.penbold(value)

    def on_fillmode(self):
        fillmode = ""
        if self.radioButtonFillDraw.isChecked():
            fillmode = "draw"
        elif self.radioButtonFillBase.isChecked():
            fillmode = "base"
        else:
            fillmode = "none"

        if self.checkBoxFill.isChecked():
            self.painter_widget.setfill(True, fillmode)
        else:
            self.painter_widget.setfill(False, "none")

    def on_previeweffect(self):
        if self.base_image is not None:
            self.base_image = self.qimage_to_pil(self.painter_widget.pixmap.toImage())
            self.mask_image = self.qimage_to_pil(self.painter_widget.pixmap_mask.toImage())
            cnet_type = self.cnet_checkpoint[1]
            _, _, image = self.imagegen.process_image(
                image_in=self.base_image,
                cnet_type=cnet_type,
                low_threshold=self.spinBoxThLow.value(),
                high_threshold=self.spinBoxThHigh.value(),
                blur=self.checkBoxBlur.isChecked(), 
                quant=self.checkBoxQuant.isChecked(),
                bayer=self.checkBoxBayer.isChecked(),
                ksize=self.spinBoxBlur.value(), 
                num_colors=self.spinBoxQuant.value(),
                image_mask=self.mask_image,
                openpose_alias=self.comboBoxAlias.currentText(),
                radius=self.spinBoxRadius.value(),
                thickness=self.spinBoxThickness.value(),
                kpt_thr=self.dspinBoxKptthr.value(),
                shuffle_k=self.spinBoxShufflek.value(),
                block=self.checkBoxBlock.isChecked(),
                block_k=self.spinBoxBlockk.value()
            )

            if cnet_type == 'inpaint':
                image = Image.fromarray((image.squeeze().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

            pixmap = QPixmap.fromImage(self.pilimage_to_qimage(image))
            
            dialog = ImagePreviewDialog()
            dialog.show_image(pixmap)
            dialog.exec()
    
    def on_cnetcp_text_changed(self, text):
        self.loading = True
        self.labelstatus.setText("Loading Model...")
        checkpoint = [cp for cp in self.cnet_checkpoints if cp[1] == text][0]
        self.cnet_checkpoint = checkpoint if checkpoint else self.cnet_checkpoints[0]

        stackwidget = [sw for sw in self.cnet_stackwidgets if sw[0] == text][0]
        self.cnet_stackwidget = stackwidget if stackwidget else self.cnet_stackwidget[0]
        self.stackedWidgetCnet.setCurrentIndex(self.cnet_stackwidget[1])

        self.reload_controlnet_run.start()
            
    def on_checkcanny(self):
        if self.checkBoxCanny.isChecked():
            self.checkBoxMask.setCheckState(Qt.CheckState.Unchecked)
            self.loading = True
            self.labelstatus.setText("Loading Model...")
            self.reload_controlnet_canny_run.start()

    def on_checkmask(self):
        if self.checkBoxMask.isChecked():
            self.checkBoxCanny.setCheckState(Qt.CheckState.Unchecked)
            self.loading = True
            self.labelstatus.setText("Loading Model...")
            self.reload_controlnet_mask_run.start()
    
    def on_drawmodefree(self):
        self.painter_widget.drawmode("free")
    
    def on_drawmodecircle(self):
        self.painter_widget.drawmode("circle")
    
    def on_drawmodeellipse(self):
        self.painter_widget.drawmode("ellipse")

    def on_drawmodeline(self):
        self.painter_widget.drawmode("line")
    
    def on_drawmodesquare(self):
        self.painter_widget.drawmode("square")

    def on_drawmodeerase(self):
        self.painter_widget.drawmode("erase")

    def reload_complete(self):
        self.loading = False
        self.labelstatus.setText("")

    def load_cnet_checkpoints(self):
        conf_controlnet = self.system.conf_system_controlnet()

        cnet_checkpoints = [(path, key) for key, path in conf_controlnet.items()]

        return cnet_checkpoints

    def set_color(self, color: QColor = Qt.GlobalColor.black):

        self.color = color
        # Create color icon
        pix_icon = QPixmap(32, 32)
        pix_icon.fill(self.color)

        self.color_action.setIcon(QIcon(pix_icon))
        self.painter_widget.pen.setColor(self.color)
        self.color_action.setText(QColor(self.color).name())

    def pilimage_to_qimage(self, pimg):
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        pimg.save(buffer, "BMP")
        qimg = QImage()
        qimg.loadFromData(buffer.data().data(), "BMP")
        return qimg
    
    def qimage_to_pil(self, qimg):
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        qimg.save(buffer, "BMP")
        fp = io. BytesIO()
        fp.write(buffer.data().data())
        buffer.close()
        fp.seek(0)
        return Image.open(fp)

def main() -> None:
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
