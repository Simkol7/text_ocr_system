"""ROI裁剪模块：按检测框裁剪文本区域，含有效性校验"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, RecognitionError

@handle_ocr_exception
def crop_roi(img: np.ndarray, box: list) -> np.ndarray:
    """
    裁剪文本ROI区域：检测框坐标校验+边界修正

    Args:
        img: 预处理后的灰度图
        box: 检测框（[x, y, w, h]）

    Returns:
        np.ndarray: 裁剪后的ROI图像
    """
    if len(img.shape) != 2:
        raise RecognitionError(f"ROI裁剪仅支持灰度图，当前维度：{img.shape}")

    x, y, w, h = box
    h_img, w_img = img.shape

    # 边界校验（防止越界）
    x = max(0, x)
    y = max(0, y)
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)

    if x2 <= x or y2 <= y:
        raise RecognitionError(f"检测框越界，无法裁剪：{box}（图像尺寸：{w_img}x{h_img}）")

    roi_img = img[y:y2, x:x2]
    logger.info(f"ROI裁剪完成，检测框：{box}，ROI尺寸：{roi_img.shape}")
    return roi_img