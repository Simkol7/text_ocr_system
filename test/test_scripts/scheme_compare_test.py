"""方案A/B对比测试：在同一张图像上分别运行方案A和方案B，量化对比效果"""
import os
import sys

from algorithm.core import logger, config_manager
from algorithm.input_module.image_loader import load_image
from algorithm.preprocess_module.preprocess_core import run_preprocess
from algorithm.ocr_scheduler import run_ocr
from test.test_scripts.batch_test import calculate_metrics
from test.test_cases.test_data import load_annotation

# 获取当前脚本的路径
current_script_path = os.path.abspath(__file__)
# 向上回溯3级，找到项目根目录（text_ocr_system/）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# 将项目根目录添加到sys.path，让Python能找到algorithm模块
sys.path.insert(0, project_root)


def compare_schemes_on_image(img_path: str, annotation_path: str | None = None) -> dict:
    """
    在同一张图像上分别运行方案A和方案B，比较检测框数量和识别指标

    Args:
        img_path: 测试图像路径
        annotation_path: 标注文件路径（可选）

    Returns:
        dict: 方案A/B的指标对比
    """
    logger.info(f"===== 方案A/B对比测试：{img_path} =====")

    # 加载图像和标注
    img, scale_factor, success = load_image(img_path)
    if not success:
        logger.error("图像加载失败，对比测试终止")
        return {}

    annotation_map = load_annotation(annotation_path) if annotation_path else {}
    annotation = annotation_map.get(img_path, [])

    # 统一预处理（避免多次重复计算）
    processed_img, process_log = run_preprocess(img)
    bin_img = process_log["bin_img"]

    # 方案A
    config_manager.config["active_scheme"] = "scheme_a"
    boxes_a, results_a, scheme_used_a = run_ocr(
        original_img=img,
        processed_img=processed_img,
        bin_img=bin_img,
        scale_factor=scale_factor,
    )
    pred_a = [r["text"] for r in results_a]
    metrics_a = calculate_metrics(pred_a, annotation)

    # 方案B
    config_manager.config["active_scheme"] = "scheme_b"
    boxes_b, results_b, scheme_used_b = run_ocr(
        original_img=img,
        processed_img=processed_img,
        bin_img=bin_img,
        scale_factor=scale_factor,
    )
    pred_b = [r["text"] for r in results_b]
    metrics_b = calculate_metrics(pred_b, annotation)

    comparison = {
        "scheme_a": {
            "scheme_used": scheme_used_a,
            "num_boxes": len(boxes_a),
            "metrics": metrics_a,
        },
        "scheme_b": {
            "scheme_used": scheme_used_b,
            "num_boxes": len(boxes_b),
            "metrics": metrics_b,
        },
        "improvement_of_b_over_a": {
            "precision": metrics_b["precision"] - metrics_a["precision"],
            "recall": metrics_b["recall"] - metrics_a["recall"],
            "f1": metrics_b["f1"] - metrics_a["f1"],
            "num_boxes": len(boxes_b) - len(boxes_a),
        },
    }

    logger.info(f"方案A/B对比结果：{comparison}")
    logger.info("===== 方案A/B对比测试完成 =====")
    return comparison


if __name__ == "__main__":
    # 默认使用挑战集中的一张图像进行对比
    test_img_path = os.path.join(
        project_root, "test/test_cases/test_images/challenge/test_low_light.jpg"
    )
    annotation_path = os.path.join(project_root, "test/test_cases/annotation.json")

    if os.path.exists(test_img_path):
        compare_schemes_on_image(test_img_path, annotation_path)
    else:
        logger.error(f"对比测试图像不存在，请检查路径：{test_img_path}")

