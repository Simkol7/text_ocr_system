"""Tesseract调用模块：动态配置PSM/语言，适配不同文本类型"""
import cv2
import numpy as np
import pytesseract
from algorithm.core import logger, handle_ocr_exception, RecognitionError, get_tesseract_path, get_config

# 设置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()

@handle_ocr_exception
def call_tesseract(roi_img: np.ndarray, psm_type: str = "single_line") -> tuple[str, float]:
    """
    调用Tesseract识别文本，返回识别结果+置信度

    Args:
        roi_img: 优化后的ROI图像
        psm_type: PSM模式类型（single_line/paragraph/single_char）

    Returns:
        tuple: (识别文本, 平均置信度)
    """
    if len(roi_img.shape) != 2:
        raise RecognitionError(f"Tesseract仅支持灰度/二值图，当前维度：{roi_img.shape}")

    # 获取PSM配置
    psm_mapping = get_config("recognition.psm_mapping")
    lang = get_config("recognition.lang")

    if psm_type not in psm_mapping:
        psm_type = "single_line"
        logger.warning(f"未知PSM类型：{psm_type}，使用默认值single_line")

    psm = psm_mapping[psm_type]
    config = f"--psm {psm} -l {lang}"

    # 调用Tesseract（获取详细结果含置信度）
    try:
        details = pytesseract.image_to_data(
            roi_img, config=config, output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        raise RecognitionError(f"Tesseract调用失败：{str(e)}")

    # 提取有效文本和置信度
    text = ""
    confidences = []
    for i, conf in enumerate(details["conf"]):
        if conf != -1:  # -1表示无文本
            text += details["text"][i] + " "
            confidences.append(conf)

    text = text.strip()
    avg_conf = np.mean(confidences) if confidences else 0.0

    logger.info(f"Tesseract识别完成，PSM：{psm}，文本：{text}，置信度：{avg_conf:.2f}")
    return text, avg_conf