"""优化效果对比测试：测试CLAHE/倾斜校正等优化点的提升效果"""
import os
import sys
import cv2
import numpy as np
from algorithm.core import logger, config_manager
from algorithm.input_module.image_loader import load_image
from algorithm.preprocess_module.preprocess_core import run_preprocess
from algorithm.detection_module.detection_core import run_detection
from algorithm.recognition_module.recognition_core import run_recognition
from test.test_scripts.batch_test import calculate_metrics
from test.test_cases.test_data import load_annotation

# 获取当前脚本的路径
current_script_path = os.path.abspath(__file__)
# 向上回溯3级，找到项目根目录（text_ocr_system/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# 将项目根目录添加到sys.path，让Python能找到algorithm模块
sys.path.insert(0, project_root)

def test_optimization_effect(img_path: str, annotation_path: str = None) -> dict:
    """
    测试优化点效果：对比开启/关闭优化的识别指标

    Args:
        img_path: 测试图像路径
        annotation_path: 标注文件路径（可选）

    Returns:
        dict: 优化前后的指标对比
    """
    logger.info(f"===== 测试优化效果：{img_path} =====")

    # 加载图像和标注
    img, scale_factor, _ = load_image(img_path)
    annotation = load_annotation(annotation_path)[img_path] if annotation_path else []

    # 1. 关闭优化（仅基础预处理：灰度+二值化）
    logger.info("测试：关闭所有优化（仅灰度+二值化）")
    original_config = config_manager.get_config("preprocess")
    config_manager.config["preprocess"]["steps"] = ["gray", "binarization"]

    processed_img_no_opt, process_log_no_opt = run_preprocess(img)
    bin_img_no_opt = process_log_no_opt["bin_img"]
    boxes_no_opt, _ = run_detection(bin_img_no_opt, scale_factor)
    results_no_opt = run_recognition(processed_img_no_opt, boxes_no_opt)
    pred_no_opt = [r["text"] for r in results_no_opt]
    metrics_no_opt = calculate_metrics(pred_no_opt, annotation)

    # 2. 开启优化（全预处理步骤）
    logger.info("测试：开启所有优化（CLAHE+降噪+形态学+倾斜校正）")
    config_manager.config["preprocess"]["steps"] = original_config["steps"] if "steps" in original_config else [
        "gray", "clahe", "noise_removal", "binarization", "morphology", "orientation_fix"
    ]

    processed_img_opt, process_log_opt = run_preprocess(img)
    bin_img_opt = process_log_opt["bin_img"]
    boxes_opt, _ = run_detection(bin_img_opt, scale_factor)
    results_opt = run_recognition(processed_img_opt, boxes_opt)
    pred_opt = [r["text"] for r in results_opt]
    metrics_opt = calculate_metrics(pred_opt, annotation)

    # 对比结果
    comparison = {
        "no_optimization": metrics_no_opt,
        "with_optimization": metrics_opt,
        "improvement": {
            "precision": metrics_opt["precision"] - metrics_no_opt["precision"],
            "recall": metrics_opt["recall"] - metrics_no_opt["recall"],
            "f1": metrics_opt["f1"] - metrics_no_opt["f1"]
        }
    }

    logger.info(f"优化效果对比：{comparison}")
    logger.info("===== 优化效果测试完成 =====")
    return comparison

if __name__ == "__main__":
    # 替换为你的测试图像路径
    test_img_path = "./test/test_cases/test_images/challenge/test_low_light.jpg"
    # 替换为你的标注文件路径（可选）
    annotation_path = "./test/test_cases/annotation.json"
    if os.path.exists(test_img_path):
        test_optimization_effect(test_img_path, annotation_path)
    else:
        logger.error("测试图像不存在，请先放入test/test_cases/test_images/challenge/目录")