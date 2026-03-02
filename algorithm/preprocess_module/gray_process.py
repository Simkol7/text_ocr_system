"""灰度化模块：将彩色图像转为灰度图，含有效性校验"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError

@handle_ocr_exception
def to_gray(img: np.ndarray) -> np.ndarray:
    """
    图像灰度化：彩色图转灰度图，灰度图直接返回

    Args:
        img: OpenCV格式图像（BGR）

    Returns:
        np.ndarray: 灰度图像（单通道）
    """
    if len(img.shape) == 2:
        logger.info("图像已是灰度图，无需处理")
        return img

    if len(img.shape) != 3 or img.shape[2] != 3:
        raise PreprocessError(f"图像通道数异常（{img.shape}），无法灰度化")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.info("图像灰度化完成，尺寸：{}".format(gray_img.shape))
    return gray_img