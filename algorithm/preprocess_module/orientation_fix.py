"""倾斜校正模块：基于最小外接矩形计算倾斜角度，自动校正"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def fix_orientation(bin_img: np.ndarray, gray_img: np.ndarray) -> np.ndarray:
    """
    倾斜校正：基于二值图轮廓计算倾斜角度，对灰度图进行旋转校正

    Args:
        bin_img: 二值化图像（用于轮廓提取）
        gray_img: 灰度图像（需要校正的图像）

    Returns:
        np.ndarray: 校正后的灰度图
    """
    if len(bin_img.shape) != 2 or len(gray_img.shape) != 2:
        raise PreprocessError("倾斜校正仅支持二值图+灰度图输入")

    # 提取轮廓（仅提取最外层轮廓）
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("未检测到轮廓，跳过倾斜校正")
        return gray_img

    # 修复：处理单轮廓场景，避免np.vstack报错
    if len(contours) == 1:
        all_contours = contours[0]
    else:
        all_contours = np.vstack(contours[i] for i in range(len(contours)))

    # 计算最小外接矩形
    rect = cv2.minAreaRect(all_contours)
    angle = rect[-1]  # 倾斜角度

    # 角度校正（OpenCV计算的角度需要转换）
    max_angle = get_config("preprocess.orientation_fix.max_angle")
    angle_threshold = get_config("preprocess.orientation_fix.angle_threshold")

    if angle < -45:
        angle += 90
    if abs(angle) < angle_threshold or abs(angle) > max_angle:
        logger.info(f"倾斜角度{angle:.2f}°（阈值{angle_threshold}°），跳过校正")
        return gray_img

    # 旋转图像（保持尺寸，黑色填充）
    h, w = gray_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    fixed_img = cv2.warpAffine(
        gray_img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )

    logger.info(f"倾斜校正完成，校正角度：{angle:.2f}°")
    return fixed_img