"""降噪模块：中值滤波去除椒盐噪点，可选高斯滤波"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def remove_noise(gray_img: np.ndarray, method: str = "median") -> np.ndarray:
    """
    图像降噪：默认中值滤波（适配文本图像）

    Args:
        gray_img: 灰度图像
        method: 降噪方法（median/ gaussian）

    Returns:
        np.ndarray: 降噪后的图像
    """
    if len(gray_img.shape) != 2:
        raise PreprocessError(f"降噪仅支持灰度图，当前维度：{gray_img.shape}")

    kernel_size = get_config("preprocess.noise_removal.kernel_size")
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保核尺寸为奇数
        logger.warning(f"降噪核尺寸需为奇数，已修正为{kernel_size}")

    if method == "median":
        denoised_img = cv2.medianBlur(gray_img, kernel_size)
    elif method == "gaussian":
        denoised_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    else:
        raise PreprocessError(f"不支持的降噪方法：{method}，仅支持median/gaussian")

    logger.info(f"图像降噪完成，方法：{method}，核尺寸：{kernel_size}")
    return denoised_img