"""结果格式化模块：将检测/识别结果转换为标准化字典格式"""
import time
from datetime import datetime
from algorithm.core import logger, handle_ocr_exception, OutputError


@handle_ocr_exception
def format_result(
    img_path: str,
    boxes: list,
    recognition_results: list,
    orientation_angle: float | None = None,
) -> dict:
    """
    将检测框和识别结果格式化为标准化字典

    Args:
        img_path: 原始图像路径
        boxes: 检测框列表（每个元素为[x, y, w, h]）
        recognition_results: 识别结果列表（兼容字典/列表/元组/字符串类型）
        orientation_angle: 倾斜校正角度（单位：度），无校正时为0或None

    Returns:
        dict: 标准化识别结果
    """
    if not isinstance(boxes, list) or not isinstance(recognition_results, list):
        raise OutputError("检测框/识别结果必须为列表类型")

    # 调试：打印识别结果的类型和内容，彻底明确结构
    logger.info(f"识别结果列表长度：{len(recognition_results)}")
    for idx, res in enumerate(recognition_results):
        logger.info(f"第{idx + 1}个识别结果 - 类型：{type(res)}，内容：{res}")

    # 过滤空识别结果（兼容所有类型）
    valid_results = []
    for idx, (box, res) in enumerate(zip(boxes, recognition_results)):
        text = ""
        confidence = 0.0

        # 核心修复：兼容不同类型的识别结果
        if isinstance(res, dict):
            # 情况1：res是字典（最常见，比如{"text": "...", "confidence": ...}）
            text = res.get("text", "").strip() if res.get("text") else ""
            confidence = round(res.get("confidence", 0.0), 2)
        elif isinstance(res, (list, tuple)):
            # 情况2：res是列表/元组（比如["=", 54.0]）
            text = res[0].strip() if len(res) >= 1 else ""
            confidence = round(res[1], 2) if len(res) >= 2 else 0.0
        elif isinstance(res, str):
            # 情况3：res只是纯文本（无置信度）
            text = res.strip()
            confidence = 0.0
        else:
            # 情况4：未知类型，打印警告并跳过
            logger.warning(f"第{idx + 1}个识别结果类型不支持：{type(res)}，内容：{res}")
            continue

        # 只保留有效结果
        if text and confidence >= 0:
            valid_results.append({
                "x": box[0],
                "y": box[1],
                "width": box[2],
                "height": box[3],
                "text": text,
                "confidence": confidence
            })
        else:
            logger.warning(f"第{idx + 1}个识别结果无效：text={text}，confidence={confidence}")

    # 构建标准化结果字典
    formatted_result = {
        "img_path": img_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_boxes": len(boxes),
        "valid_recognitions": len(valid_results),
        "orientation_angle": float(orientation_angle) if orientation_angle is not None else 0.0,
        "recognitions": valid_results,
    }

    logger.info("识别结果标准化完成")
    return formatted_result