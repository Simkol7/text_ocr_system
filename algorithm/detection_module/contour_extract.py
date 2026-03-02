"""轮廓提取模块：从二值图提取文本轮廓，含层级筛选"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError

@handle_ocr_exception
def extract_contours(bin_img: np.ndarray) -> list:
    """
    提取文本轮廓：仅提取外层轮廓，过滤极小轮廓

    Args:
        bin_img: 二值化图像（黑白）

    Returns:
        list: 轮廓列表（cv2轮廓格式）
    """
    if len(bin_img.shape) != 2:
        raise DetectionError(f"轮廓提取仅支持二值图，当前维度：{bin_img.shape}")

    # 提取轮廓（RETR_EXTERNAL：仅最外层轮廓）
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise DetectionError("未检测到任何轮廓")

    logger.info(f"轮廓提取完成，原始轮廓数：{len(contours)}")
    return contours