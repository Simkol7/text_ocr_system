"""检测框坐标还原：适配预处理的缩放/旋转，还原到原始图像坐标"""
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError

@handle_ocr_exception
def restore_boxes(boxes: list, scale_factor: float = 1.0) -> list:
    """
    还原检测框坐标：将预处理后图像的框坐标还原到原始图像

    Args:
        boxes: 预处理后图像的检测框列表（[x, y, w, h]）
        scale_factor: 图像缩放系数（预处理时的缩放比例）

    Returns:
        list: 还原后的检测框列表
    """
    if not boxes:
        raise DetectionError("检测框列表为空，无法还原")
    if scale_factor <= 0:
        raise DetectionError(f"缩放系数非法：{scale_factor}")

    restored_boxes = []
    for box in boxes:
        x, y, w, h = box
        # 还原坐标（除以缩放系数，取整）
        x_restored = int(x / scale_factor)
        y_restored = int(y / scale_factor)
        w_restored = int(w / scale_factor)
        h_restored = int(h / scale_factor)
        restored_boxes.append([x_restored, y_restored, w_restored, h_restored])

    logger.info(f"检测框坐标还原完成，缩放系数：{scale_factor}")
    return restored_boxes