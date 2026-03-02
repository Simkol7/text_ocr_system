"""批量图像加载模块：支持文件夹批量读取，含进度条/异常跳过"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from algorithm.core import logger, handle_ocr_exception, InputError, get_config
from .image_loader import load_image

@handle_ocr_exception
def load_batch_images(folder_path: str, scale_factor: float = 1.0) -> dict:
    """
    批量加载文件夹内的所有图像，跳过无效文件

    Args:
        folder_path: 图像文件夹路径
        scale_factor: 缩放系数

    Returns:
        dict: 加载结果（key：图像路径，value：(图像, 缩放系数, 是否成功)）
    """
    if not os.path.exists(folder_path):
        raise InputError(f"文件夹路径不存在：{folder_path}")
    if not os.path.isdir(folder_path):
        raise InputError(f"路径不是文件夹：{folder_path}")

    valid_formats = get_config("input.valid_formats")
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[-1].lower().lstrip(".") in valid_formats
    ]

    if not image_files:
        raise InputError(f"文件夹{folder_path}内无有效图像（支持格式：{','.join(valid_formats)}）")

    logger.info(f"开始批量加载图像，共{len(image_files)}个文件")
    batch_results = {}

    # 进度条展示
    for img_path in tqdm(image_files, desc="Loading Images"):
        try:
            img, scale, success = load_image(img_path, scale_factor)
            batch_results[img_path] = (img, scale, success)
        except Exception as e:
            logger.error(f"加载图像{img_path}失败：{str(e)}，跳过")
            batch_results[img_path] = (None, scale_factor, False)

    # 统计加载结果
    success_count = len([v for v in batch_results.values() if v[2]])
    logger.info(f"批量加载完成，成功：{success_count}/{len(image_files)}")
    return batch_results