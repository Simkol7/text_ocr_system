"""检测流程整合：从二值图到最终检测框，含可视化标注"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError
from .contour_extract import extract_contours
from .contour_filter import filter_contours
from .contour_merge import merge_contours
from .box_restore import restore_boxes

@handle_ocr_exception
def run_detection(bin_img: np.ndarray, scale_factor: float = 1.0) -> tuple[list, np.ndarray]:
    """
    执行完整文本检测流程，返回检测框+标注图像

    Args:
        bin_img: 二值化图像
        scale_factor: 图像缩放系数（用于还原坐标）

    Returns:
        tuple: (还原后的检测框列表, 标注检测框的图像)
    """
    logger.info("开始文本检测流程")

    # 1. 轮廓提取
    contours = extract_contours(bin_img)
    # 2. 轮廓筛选
    filtered_contours = filter_contours(contours)
    if not filtered_contours:
        raise DetectionError("筛选后无有效轮廓，检测失败")
    # 3. 轮廓合并（生成检测框）
    boxes = merge_contours(filtered_contours)
    # 4. 坐标还原
    restored_boxes = restore_boxes(boxes, scale_factor)

    # 5. 可视化标注（在二值图上绘制检测框）
    vis_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    logger.info(f"检测流程完成，最终检测框数：{len(restored_boxes)}")
    return restored_boxes, vis_img