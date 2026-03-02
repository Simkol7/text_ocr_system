"""单模块测试：单独验证预处理/检测/识别模块"""
import os
import sys
import cv2
import numpy as np
from algorithm.core import logger, config_manager
from algorithm.input_module.image_loader import load_image
from algorithm.preprocess_module.preprocess_core import run_preprocess
from algorithm.detection_module.detection_core import run_detection
from algorithm.recognition_module.recognition_core import run_recognition

# 获取当前脚本的路径
current_script_path = os.path.abspath(__file__)
# 向上回溯3级，找到项目根目录（text_ocr_system/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# 将项目根目录添加到sys.path，让Python能找到algorithm模块
sys.path.insert(0, project_root)

def test_preprocess(img_path: str):
    """测试预处理模块"""
    logger.info("===== 测试预处理模块 =====")
    img, scale, _ = load_image(img_path)
    processed_img, process_log = run_preprocess(img)
    logger.info(f"预处理日志：{process_log}")
    cv2.imwrite("./test/test_results/preprocess_test.jpg", processed_img)
    logger.info("预处理测试完成，结果保存到test/test_results/preprocess_test.jpg")

def test_detection(img_path: str):
    """测试检测模块"""
    logger.info("===== 测试检测模块 =====")
    img, scale, _ = load_image(img_path)
    processed_img, process_log = run_preprocess(img)
    bin_img = process_log["bin_img"]
    boxes, vis_img = run_detection(bin_img, scale)
    cv2.imwrite("./test/test_results/detection_test.jpg", vis_img)
    logger.info(f"检测测试完成，检测框数：{len(boxes)}，结果保存到test/test_results/detection_test.jpg")

def test_recognition(img_path: str):
    """测试识别模块"""
    logger.info("===== 测试识别模块 =====")
    img, scale, _ = load_image(img_path)
    processed_img, process_log = run_preprocess(img)
    bin_img = process_log["bin_img"]
    boxes, _ = run_detection(bin_img, scale)
    results = run_recognition(processed_img, boxes)
    logger.info(f"识别测试完成，有效文本数：{len([r for r in results if r['text']])}")
    return results

if __name__ == "__main__":
    # 替换为你的测试图像路径
    test_img_path = "./test/test_cases/test_images/test.jpg"
    if os.path.exists(test_img_path):
        test_preprocess(test_img_path)
        test_detection(test_img_path)
        test_recognition(test_img_path)
    else:
        logger.error("测试图像不存在，请先放入test/test_cases/test_images/目录")