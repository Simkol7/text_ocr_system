"""测试数据集管理：标注数据加载、测试集分类"""
import os
import json
from algorithm.core import logger, handle_ocr_exception

@handle_ocr_exception
def load_annotation(annotation_path: str) -> dict:
    """
    加载人工标注数据（JSON格式）

    Args:
        annotation_path: 标注文件路径

    Returns:
        dict: 标注数据（key：图像路径，value：标注文本列表）
    """
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"标注文件不存在：{annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    logger.info(f"加载标注数据完成，共{len(annotation)}个图像标注")
    return annotation

@handle_ocr_exception
def get_test_sets(base_path: str) -> dict:
    """
    获取分类测试集（基础集/进阶集/挑战集）

    Args:
        base_path: 测试图像根目录

    Returns:
        dict: 测试集路径（key：测试集名称，value：文件夹路径）
    """
    test_sets = {
        "basic": os.path.join(base_path, "basic"),
        "advanced": os.path.join(base_path, "advanced"),
        "challenge": os.path.join(base_path, "challenge")
    }

    # 检查测试集路径（避免遍历时修改字典）
    available_sets = {}
    for name, path in test_sets.items():
        if os.path.exists(path):
            available_sets[name] = path
        else:
            logger.warning(f"测试集{name}路径不存在：{path}，已跳过")

    logger.info(f"加载测试集完成，可用测试集：{list(available_sets.keys())}")
    return available_sets