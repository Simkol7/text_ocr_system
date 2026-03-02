"""图像加载模块：路径校验、图像加载、缩放、有效性校验"""
import os
import cv2
import numpy as np
from typing import Tuple, Optional
from algorithm.core import logger, handle_ocr_exception, InputError, get_config

@handle_ocr_exception
def load_image(img_path: str, scale_factor: float = 1.0) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    加载图像并执行基础校验，支持缩放

    Args:
        img_path: 图像文件路径
        scale_factor: 缩放系数（0.5-2.0）

    Returns:
        tuple: (加载后的图像/None, 实际缩放系数, 是否成功)
    """
    # 1. 路径校验
    if not os.path.exists(img_path):
        raise InputError(f"图像路径不存在：{img_path}")
    if not os.path.isfile(img_path):
        raise InputError(f"路径不是文件：{img_path}")

    # 2. 格式校验
    valid_formats = get_config("input.valid_formats")
    file_ext = os.path.splitext(img_path)[-1].lower().lstrip(".")
    if file_ext not in valid_formats:
        raise InputError(f"图像格式不支持（{file_ext}），仅支持：{','.join(valid_formats)}")

    # 3. 加载图像
    img = cv2.imread(img_path)
    if img is None:
        raise InputError(f"图像加载失败（可能文件损坏）：{img_path}")
    logger.info(f"图像加载成功：{img_path}，原始尺寸：{img.shape[:2]}")

    # 4. 缩放系数校验+图像缩放
    scale_range = get_config("input.scale_range")
    if not (scale_range[0] <= scale_factor <= scale_range[1]):
        scale_factor = 1.0
        logger.warning(f"缩放系数{scale_factor}超出范围{scale_range}，已修正为1.0")

    if scale_factor != 1.0:
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        logger.info(f"图像缩放完成，缩放系数：{scale_factor}，新尺寸：{img.shape[:2]}")

    return img, scale_factor, True