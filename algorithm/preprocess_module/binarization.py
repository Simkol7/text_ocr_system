"""二值化模块：全局/自适应二值化，将灰度图转为黑白图"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def binarize(gray_img: np.ndarray) -> np.ndarray:
    """
    图像二值化：从配置读取二值化类型，自适应二值化优先

    Args:
        gray_img: 灰度图像

    Returns:
        np.ndarray: 二值化图像（黑白）
    """
    if len(gray_img.shape) != 2:
        raise PreprocessError(f"二值化仅支持灰度图，当前维度：{gray_img.shape}")

    bin_config = get_config("preprocess.binarization")
    bin_type = bin_config["type"]

    if bin_type == "global":
        # 全局二值化（OTSU自动阈值）
        _, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif bin_type == "adaptive":
        # 自适应二值化（适配局部光照不均）
        block_size = bin_config["adaptive_block_size"]
        c = bin_config["adaptive_c"]
        if block_size % 2 == 0:
            block_size += 1
            logger.warning(f"自适应二值化块尺寸需为奇数，已修正为{block_size}")
        bin_img = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c
        )
    else:
        raise PreprocessError(f"不支持的二值化类型：{bin_type}，仅支持global/adaptive")

    logger.info(f"图像二值化完成，类型：{bin_type}")
    return bin_img