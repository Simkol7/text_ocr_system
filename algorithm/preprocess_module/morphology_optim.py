"""形态学优化模块：膨胀/腐蚀/开运算/闭运算修复字符边缘"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def morphology_optimize(bin_img: np.ndarray) -> np.ndarray:
    """
    形态学优化：闭运算填充字符内部孔洞，开运算去除小噪点

    Args:
        bin_img: 二值化图像

    Returns:
        np.ndarray: 形态学优化后的二值图
    """
    if len(bin_img.shape) != 2:
        raise PreprocessError(f"形态学操作仅支持二值图，当前维度：{bin_img.shape}")

    kernel_size = get_config("preprocess.morphology.kernel_size")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # 先闭运算（填充孔洞），再开运算（去除噪点）
    close_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    morph_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)

    logger.info(f"形态学优化完成，核尺寸：{kernel_size}")
    return morph_img