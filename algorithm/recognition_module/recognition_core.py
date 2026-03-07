"""识别流程整合：关联检测框+ROI裁剪+识别，输出带坐标的文本结果"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, RecognitionError
from .roi_crop import crop_roi
from .roi_optim import optimize_roi
from .tesseract_call import call_tesseract


def _get_intelligent_psm(box: list) -> str:
    """
    智能判断逻辑：根据检测框几何特征返回对应的 PSM 类型

    逻辑依据：
    1. 长宽比极高 (w/h > 2.0) -> 单行文本 (single_line, PSM 7)
    2. 长宽比接近 1 且 面积较小 -> 单个字符 (single_char, PSM 10)
    3. 其他情况（如大面积块状） -> 段落/多行 (paragraph, PSM 3)
    """
    x, y, w, h = box
    aspect_ratio = w / h if h > 0 else 0
    area = w * h

    # 1. 单个字符判断：面积较小的细长/近方块区域都视为单字符
    #    例如本项目中常见的小字宽高在 10~40 像素范围内
    if area < 2500:
        return "single_char"

    # 2. 单行文本判断：宽度明显大于高度
    if aspect_ratio >= 1.5:
        return "single_line"

    # 3. 默认返回段落模式
    return "paragraph"


@handle_ocr_exception
def run_recognition(img: np.ndarray, boxes: list) -> list:
    """
    执行完整文本识别流程，包含智能 PSM 选择

    Args:
        img: 预处理后的灰度图
        boxes: 还原后的检测框列表（[x, y, w, h]）

    Returns:
        list: 识别结果列表（无检测框时返回空列表）
    """
    if not boxes:
        logger.info("检测框列表为空，判定为无文本场景，跳过识别并返回空结果")
        return []
    if len(img.shape) != 2:
        raise RecognitionError(f"识别仅支持灰度图，当前维度：{img.shape}")

    logger.info(f"开始文本识别流程，检测框数：{len(boxes)}")
    results = []

    for box in boxes:
        try:
            # --- 智能 PSM 选择优化点 ---
            psm_type = _get_intelligent_psm(box)
            logger.debug(f"检测框 {box} 自动匹配 PSM 类型: {psm_type}")

            # 1. ROI 裁剪
            roi_img = crop_roi(img, box)

            # 2. ROI 优化（二值化/膨胀）
            roi_optim = optimize_roi(roi_img)

            # 3. 调用 Tesseract（传入动态确定的 psm_type）
            text, confidence = call_tesseract(roi_optim, psm_type=psm_type)

            # 保存结果
            results.append({
                "box": box,
                "text": text,
                "confidence": round(confidence, 2),
                "psm_mode": psm_type  # 记录 PSM 模式，方便后续性能分析
            })
        except Exception as e:
            logger.error(f"检测框 {box} 识别失败：{str(e)}")
            results.append({
                "box": box,
                "text": "",
                "confidence": 0.0,
                "psm_mode": "failed"
            })

    logger.info(f"识别流程完成，有效识别文本数：{len([r for r in results if r['text']])}")
    return results