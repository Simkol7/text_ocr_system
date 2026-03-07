"""OCR调度模块：在方案A（OpenCV+Tesseract）与方案B（EAST+CRNN）之间无感切换

本模块作为上层（如UI、测试脚本）与底层算法之间的唯一桥梁：
- 通过active_scheme配置决定当前使用方案A还是方案B
- 对外暴露统一的run_ocr接口，返回与方案A完全一致的结果结构（boxes + recognition_results）
"""
from typing import List, Tuple

import numpy as np
import cv2

from algorithm.core import logger, get_config, handle_ocr_exception, DetectionError, RecognitionError
from algorithm.detection_module.detection_core import run_detection as run_detection_a
from algorithm.detection_module.east_detector import get_east_detector
from algorithm.recognition_module.recognition_core import run_recognition as run_recognition_a
from algorithm.recognition_module.crnn_recognizer import get_crnn_recognizer
from algorithm.recognition_module.roi_crop import crop_roi


@handle_ocr_exception
def _run_scheme_a(
    processed_img: np.ndarray,
    bin_img: np.ndarray,
    scale_factor: float,
) -> Tuple[List[List[int]], List[dict]]:
    """方案A：预处理二值图 + 轮廓检测 + Tesseract识别"""
    boxes, _ = run_detection_a(bin_img, scale_factor)
    recognition_results = run_recognition_a(processed_img, boxes)
    return boxes, recognition_results


@handle_ocr_exception
def _run_scheme_b(
    original_img: np.ndarray,
    scale_factor: float,
) -> Tuple[List[List[int]], List[dict]]:
    """方案B：EAST检测 + CRNN识别

    为了与方案A输出结构兼容，这里手工构造与run_recognition类似的结果字典：
    [
        {"box": [x,y,w,h], "text": str, "confidence": float, "psm_mode": "crnn"}
    ]
    """
    east = get_east_detector()
    crnn = get_crnn_recognizer()

    # EAST基于当前输入尺寸工作，scale_factor在方案B中只用于保持接口一致，这里不再二次还原
    boxes = east.detect(original_img)
    results: List[dict] = []

    if not boxes:
        logger.info("方案B：EAST未检测到文本区域，将返回空结果")
        return [], []

    gray_for_roi = (
        cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        if len(original_img.shape) == 3
        else original_img
    )

    for box in boxes:
        try:
            roi_img = crop_roi(gray_for_roi, box)
            # 转为BGR再送给CRNN，保持接口一致
            roi_bgr = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
            text, conf = crnn.recognize(roi_bgr)
            results.append(
                {
                    "box": box,
                    "text": text,
                    "confidence": round(float(conf), 4),
                    "psm_mode": "crnn",  # 兼容字段，占位表示方案B
                }
            )
        except Exception as e:
            logger.error(f"方案B：检测框{box}识别失败：{str(e)}")
            results.append(
                {
                    "box": box,
                    "text": "",
                    "confidence": 0.0,
                    "psm_mode": "failed",
                }
            )

    return boxes, results


@handle_ocr_exception
def run_ocr(
    original_img: np.ndarray,
    processed_img: np.ndarray,
    bin_img: np.ndarray,
    scale_factor: float,
) -> Tuple[List[List[int]], List[dict], str]:
    """
    统一OCR调度入口：根据active_scheme在方案A/B之间切换

    Args:
        original_img: 原始BGR图像（经过load_image后的图像）
        processed_img: 预处理后的灰度图（供方案A识别使用）
        bin_img: 预处理阶段生成的二值图（供方案A检测使用）
        scale_factor: 缩放系数（方案A用于坐标还原，方案B仅保持接口一致）

    Returns:
        tuple:
            - boxes: 检测框列表
            - recognition_results: 识别结果列表（结构与方案A一致）
            - scheme_used: 实际使用的方案标记（"scheme_a" 或 "scheme_b"）
    """
    active_scheme = get_config("active_scheme")
    logger.info(f"OCR调度器启动，当前配置方案：{active_scheme}")

    # 优先尝试配置指定方案；若方案B初始化失败，则自动回退到方案A
    if active_scheme == "scheme_b":
        try:
            boxes, results = _run_scheme_b(original_img, scale_factor)
            return boxes, results, "scheme_b"
        except (DetectionError, RecognitionError) as e:
            logger.error(f"方案B执行失败，将自动回退到方案A：{str(e)}")

    # 默认或回退：方案A
    boxes, results = _run_scheme_a(processed_img, bin_img, scale_factor)
    return boxes, results, "scheme_a"

