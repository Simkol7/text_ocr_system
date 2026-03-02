"""CLAHE增强模块：解决光照不均问题，提升文本对比度"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def clahe_enhance(gray_img: np.ndarray) -> np.ndarray:
    """
    CLAHE自适应直方图均衡化：仅对灰度图生效

    Args:
        gray_img: 灰度图像（单通道）

    Returns:
        np.ndarray: CLAHE增强后的灰度图
    """
    if len(gray_img.shape) != 2:
        raise PreprocessError(f"CLAHE仅支持灰度图，当前图像维度：{gray_img.shape}")

    # 从配置读取CLAHE参数
    clahe_config = get_config("preprocess.clahe")
    clip_limit = clahe_config["clip_limit"]
    tile_grid_size = tuple(clahe_config["tile_grid_size"])

    # 初始化CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(gray_img)
    logger.info(f"CLAHE增强完成，参数：clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
    return clahe_img