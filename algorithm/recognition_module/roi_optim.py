"""ROI预处理模块：精细化优化裁剪后的文本区域"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, RecognitionError

@handle_ocr_exception
def optimize_roi(roi_img: np.ndarray) -> np.ndarray:
    """
    ROI精细化优化：提升小文本区域的识别率

    Args:
        roi_img: 裁剪后的ROI灰度图

    Returns:
        np.ndarray: 优化后的ROI图像
    """
    if len(roi_img.shape) != 2:
        raise RecognitionError(f"ROI优化仅支持灰度图，当前维度：{roi_img.shape}")

    # 1. 自适应阈值二值化（针对小区域）
    h, w = roi_img.shape
    block_size = min(15, max(3, h // 2, w // 2))
    if block_size % 2 == 0:
        block_size += 1
    roi_bin = cv2.adaptiveThreshold(
        roi_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 1
    )

    # 2. 轻微膨胀（增强字符边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    roi_optim = cv2.dilate(roi_bin, kernel, iterations=1)

    logger.info(f"ROI优化完成，块尺寸：{block_size}")
    return roi_optim