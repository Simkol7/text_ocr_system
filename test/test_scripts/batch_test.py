"""批量测试：文件夹内所有图像测试，计算识别准确率/召回率"""
import os
import sys
import numpy as np
from algorithm.core import logger
from algorithm.input_module.batch_loader import load_batch_images
from algorithm.preprocess_module.preprocess_core import run_preprocess
from algorithm.ocr_scheduler import run_ocr
from algorithm.output_module.result_format import format_result
from test.test_cases.test_data import load_annotation

# 获取当前脚本的路径
current_script_path = os.path.abspath(__file__)
# 向上回溯3级，找到项目根目录（text_ocr_system/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# 将项目根目录添加到sys.path，让Python能找到algorithm模块
sys.path.insert(0, project_root)
def calculate_metrics(predicted: list, annotated: list) -> dict:
    """
    计算识别准确率和召回率（简单文本匹配）

    Args:
        predicted: 预测文本列表
        annotated: 标注文本列表

    Returns:
        dict: 指标（准确率、召回率、F1值）
    """
    # 去重+转小写（简化匹配）
    pred_set = set([p.lower().strip() for p in predicted if p])
    anno_set = set([a.lower().strip() for a in annotated if a])

    if not anno_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # 计算指标
    tp = len(pred_set & anno_set)  # 真阳性
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(anno_set) if anno_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def batch_ocr_test(folder_path: str, annotation_path: str = None) -> dict:
    """
    批量OCR测试，计算整体指标

    Args:
        folder_path: 测试图像文件夹路径
        annotation_path: 标注文件路径（可选）

    Returns:
        dict: 批量测试结果+指标
    """
    logger.info("===== 开始批量OCR测试 =====")

    # 1. 加载批量图像
    batch_results = load_batch_images(folder_path)

    # 2. 加载标注数据（可选）
    annotation = load_annotation(annotation_path) if annotation_path else {}

    # 3. 批量处理
    all_metrics = []
    batch_formatted = {}

    for img_path, (img, scale_factor, success) in batch_results.items():
        if not success:
            logger.warning(f"跳过无效图像：{img_path}")
            continue

        # 单图像OCR流程（通过调度器自动选择方案A/B）
        processed_img, process_log = run_preprocess(img)
        bin_img = process_log["bin_img"]
        orientation_angle = process_log.get("orientation_angle", 0.0)
        boxes, recognition_results, scheme_used = run_ocr(
            original_img=img,
            processed_img=processed_img,
            bin_img=bin_img,
            scale_factor=scale_factor,
        )
        logger.info(f"批量测试：图像{img_path}使用方案：{scheme_used}")
        formatted_result = format_result(
            img_path, boxes, recognition_results, orientation_angle=orientation_angle
        )
        batch_formatted[img_path] = formatted_result

        # 计算指标（如果有标注）
        if img_path in annotation:
            # 使用标准化结果中的recognitions字段
            predicted = [r["text"] for r in formatted_result["recognitions"]]
            annotated = annotation[img_path]
            metrics = calculate_metrics(predicted, annotated)
            all_metrics.append(metrics)
            logger.info(f"图像{img_path}指标：{metrics}")

    # 计算整体指标
    if all_metrics:
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_f1 = np.mean([m["f1"] for m in all_metrics])
        overall_metrics = {
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1": round(avg_f1, 4)
        }
    else:
        overall_metrics = {"avg_precision": 0.0, "avg_recall": 0.0, "avg_f1": 0.0}

    logger.info(f"===== 批量OCR测试完成 =====")
    logger.info(f"整体指标：{overall_metrics}")
    return {"batch_results": batch_formatted, "overall_metrics": overall_metrics}

if __name__ == "__main__":
    # 替换为你的测试文件夹路径
    test_folder = "./test/test_cases/test_images/basic"
    # 替换为你的标注文件路径（可选）
    annotation_path = "./test/test_cases/annotation.json"
    batch_ocr_test(test_folder, annotation_path)