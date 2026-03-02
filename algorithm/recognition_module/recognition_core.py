"""识别流程整合：关联检测框+ROI裁剪+识别，输出带坐标的文本结果"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, RecognitionError
from .roi_crop import crop_roi
from .roi_optim import optimize_roi
from .tesseract_call import call_tesseract

@handle_ocr_exception
def run_recognition(img: np.ndarray, boxes: list) -> list:
    """
    执行完整文本识别流程，返回带坐标的识别结果

    Args:
        img: 预处理后的灰度图
        boxes: 还原后的检测框列表（[x, y, w, h]）

    Returns:
        list: 识别结果列表（每个元素：{"box": [x,y,w,h], "text": "", "confidence": 0.0}）
    """
    if not boxes:
        raise RecognitionError("检测框列表为空，无法识别")
    if len(img.shape) != 2:
        raise RecognitionError(f"识别仅支持灰度图，当前维度：{img.shape}")

    logger.info(f"开始文本识别流程，检测框数：{len(boxes)}")
    results = []

    for box in boxes:
        try:
            # 1. ROI裁剪
            roi_img = crop_roi(img, box)
            # 2. ROI优化
            roi_optim = optimize_roi(roi_img)
            # 3. 调用Tesseract
            text, confidence = call_tesseract(roi_optim)

            # 保存结果
            results.append({
                "box": box,
                "text": text,
                "confidence": round(confidence, 2)
            })
        except Exception as e:
            logger.error(f"检测框{box}识别失败：{str(e)}")
            results.append({
                "box": box,
                "text": "",
                "confidence": 0.0
            })

    logger.info(f"识别流程完成，有效识别文本数：{len([r for r in results if r['text']])}")
    return results