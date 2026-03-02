"""全链路测试：单张图像从输入到输出的完整OCR流程"""
import os
import sys
import cv2

# ===================== 核心修复：添加项目根目录到Python路径 =====================
# 获取当前脚本的路径
current_script_path = os.path.abspath(__file__)
# 向上回溯3级，找到项目根目录（text_ocr_system/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# 将项目根目录添加到sys.path，让Python能找到algorithm模块
sys.path.insert(0, project_root)
# ===========================================================================

from algorithm.core import logger, config_manager
from algorithm.input_module.image_loader import load_image
from algorithm.preprocess_module.preprocess_core import run_preprocess
from algorithm.detection_module.detection_core import run_detection
from algorithm.recognition_module.recognition_core import run_recognition
from algorithm.output_module.result_format import format_result
from algorithm.output_module.result_save import save_result
from algorithm.output_module.result_show import print_result


def full_ocr_test(img_path: str, save_result_flag: bool = True) -> dict:
    """
    执行完整OCR流程：加载→预处理→检测→识别→格式化→保存/打印

    Args:
        img_path: 测试图像路径
        save_result_flag: 是否保存结果

    Returns:
        dict: 标准化识别结果
    """
    logger.info("===== 开始全链路OCR测试 =====")

    # 1. 加载图像
    img, scale_factor, success = load_image(img_path)
    if not success:
        logger.error("图像加载失败，测试终止")
        return {}

    # 2. 预处理
    processed_img, process_log = run_preprocess(img)
    bin_img = process_log["bin_img"]

    # 3. 检测
    boxes, vis_img = run_detection(bin_img, scale_factor)

    # 4. 识别
    recognition_results = run_recognition(processed_img, boxes)

    # 5. 结果格式化
    formatted_result = format_result(img_path, boxes, recognition_results)

    # 6. 结果保存/打印（修改：传递project_root）
    if save_result_flag:
        save_result(formatted_result, vis_img, img_path, project_root)  # 新增project_root参数
    print_result(formatted_result)

    logger.info("===== 全链路OCR测试完成 =====")
    return formatted_result

if __name__ == "__main__":
    # 替换为你的测试图像路径（建议用绝对路径，避免相对路径问题）
    # 示例：test_img_path = "C:/Users/29707/Desktop/text_ocr_system/test/test_cases/test_images/test.jpg"
    test_img_path = os.path.join(project_root, "test/test_cases/test_images/test.jpg")

    if os.path.exists(test_img_path):
        full_ocr_test(test_img_path)
    else:
        logger.error(f"测试图像不存在，请检查路径：{test_img_path}")
        logger.info("提示：请在test/test_cases/test_images/目录下放入test.jpg测试图片")