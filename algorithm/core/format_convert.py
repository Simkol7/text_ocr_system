"""格式转换模块：OpenCV图像↔PyQt5图像格式互转，统一GUI显示"""
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

def cv2_to_qt(cv_img):
    """
    OpenCV图像（BGR）转换为PyQt5 QPixmap，适配界面显示

    Args:
        cv_img: OpenCV格式图像（np.ndarray）

    Returns:
        QPixmap: PyQt5显示用图像
    """
    if len(cv_img.shape) == 3:
        # 彩色图像：BGR → RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    else:
        # 灰度图像
        h, w = cv_img.shape
        q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(q_img)

def qt_to_cv(qt_img):
    """
    PyQt5 QImage转换为OpenCV图像（BGR）

    Args:
        qt_img: QImage格式图像

    Returns:
        np.ndarray: OpenCV格式图像
    """
    qt_img = qt_img.convertToFormat(QImage.Format_RGB888)
    width = qt_img.width()
    height = qt_img.height()
    ptr = qt_img.bits()
    ptr.setsize(qt_img.byteCount())
    cv_img = np.array(ptr).reshape(height, width, 3)
    # RGB → BGR
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img