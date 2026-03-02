"""结果展示模块：在控制台打印标准化的识别结果"""
from algorithm.core import logger, handle_ocr_exception, OutputError

@handle_ocr_exception
def print_result(formatted_result: dict) -> None:
    """
    在控制台友好打印识别结果

    Args:
        formatted_result: 标准化识别结果

    Raises:
        OutputError: 打印失败时抛出异常
    """
    if not isinstance(formatted_result, dict):
        raise OutputError("识别结果必须为字典类型")

    # 容错处理：确保关键字段存在，不存在则显示默认值
    img_path = formatted_result.get("img_path", "未知路径")
    timestamp = formatted_result.get("timestamp", "未知时间")
    total_boxes = formatted_result.get("total_boxes", 0)
    valid_recognitions = formatted_result.get("valid_recognitions", 0)
    recognitions = formatted_result.get("recognitions", [])

    # 打印汇总信息
    logger.info("\n===== 识别结果汇总 =====")
    print(f"\n===== 识别结果汇总 =====")
    print(f"识别时间：{timestamp}")
    print(f"原始图像：{img_path}")  # 核心修复：将original_image改为img_path
    print(f"总检测框数：{total_boxes}")
    print(f"有效识别数：{valid_recognitions}")

    # 打印详细识别结果
    if recognitions:
        print("\n===== 详细识别结果 =====")
        for idx, res in enumerate(recognitions, 1):
            text = res.get("text", "")
            confidence = res.get("confidence", 0.0)
            x = res.get("x", 0)
            y = res.get("y", 0)
            print(f"[{idx}] 坐标({x},{y})：{text}（置信度：{confidence:.1f}）")
    else:
        print("\n===== 详细识别结果 =====")
        print("无有效识别结果")

    logger.info("识别结果打印完成")