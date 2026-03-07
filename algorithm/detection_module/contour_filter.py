"""轮廓筛选模块：按面积、长宽比、实心度过滤无效轮廓"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError, get_config

@handle_ocr_exception
def filter_contours(contours: list) -> list:
    """
    筛选有效文本轮廓：去除过小/过长/过扁/实心度低的轮廓

    Args:
        contours: 原始轮廓列表

    Returns:
        list: 筛选后的轮廓列表
    """
    if not contours:
        logger.warning("轮廓列表为空，跳过筛选，将视为无文本场景")
        return []

    filter_config = get_config("detection.contour_filter")
    min_area = filter_config["min_area"]
    aspect_ratio_min, aspect_ratio_max = filter_config["aspect_ratio_range"]
    solidity_threshold = filter_config["solidity_threshold"]
    min_height = filter_config["min_height"]

    filtered_contours = []
    for cnt in contours:
        # 1. 面积筛选
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 2. 外接矩形筛选（长宽比、高度）
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0
        if not (aspect_ratio_min <= aspect_ratio <= aspect_ratio_max) or h < min_height:
            continue

        # 3. 实心度筛选（轮廓面积/凸包面积）
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < solidity_threshold:
            continue

        filtered_contours.append(cnt)

    logger.info(f"轮廓筛选完成，筛选后轮廓数：{len(filtered_contours)}（原始：{len(contours)}）")
    return filtered_contours