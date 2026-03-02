"""轮廓合并模块：合并相邻/重叠的文本轮廓，生成检测框"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError, get_config

@handle_ocr_exception
def merge_contours(contours: list) -> list:
    """
    合并相邻轮廓：按水平距离合并，生成文本行检测框

    Args:
        contours: 筛选后的轮廓列表

    Returns:
        list: 合并后的检测框列表（每个框：[x, y, w, h]）
    """
    if not contours:
        raise DetectionError("轮廓列表为空，无法合并")

    merge_distance = get_config("detection.merge_distance")

    # 按轮廓y坐标排序（从上到下）
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

    boxes = []
    current_box = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if current_box is None:
            current_box = [x, y, x + w, y + h]
        else:
            # 判断是否在同一行（y轴重叠/距离近）
            if abs(y - current_box[1]) < merge_distance or abs((y + h) - current_box[3]) < merge_distance:
                # 合并框：取最小x、最小y、最大x、最大y
                current_box[0] = min(current_box[0], x)
                current_box[1] = min(current_box[1], y)
                current_box[2] = max(current_box[2], x + w)
                current_box[3] = max(current_box[3], y + h)
            else:
                # 保存当前框，新建框
                boxes.append([current_box[0], current_box[1], current_box[2] - current_box[0], current_box[3] - current_box[1]])
                current_box = [x, y, x + w, y + h]

    # 保存最后一个框
    if current_box is not None:
        boxes.append([current_box[0], current_box[1], current_box[2] - current_box[0], current_box[3] - current_box[1]])

    logger.info(f"轮廓合并完成，检测框数：{len(boxes)}")
    return boxes