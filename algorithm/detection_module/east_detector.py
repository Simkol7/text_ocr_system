"""EAST文本检测模块：使用cv2.dnn加载.pb模型，输出与方案A兼容的水平框格式"""
import os
from typing import List, Tuple

import cv2
import numpy as np

from algorithm.core import logger, handle_ocr_exception, DetectionError, get_config


def _decode_east_scores(
    scores: np.ndarray,
    geometry: np.ndarray,
    score_thresh: float,
) -> Tuple[List[List[int]], List[float]]:
    """
    从EAST输出的score与geometry特征图中解析候选框，返回axis-aligned框和对应置信度
    """
    num_rows, num_cols = scores.shape[2:4]
    boxes: List[List[int]] = []
    confidences: List[float] = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0_data = geometry[0, 0, y]
        x1_data = geometry[0, 1, y]
        x2_data = geometry[0, 2, y]
        x3_data = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            score = float(scores_data[x])
            if score < score_thresh:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0

            angle = float(angles_data[x])
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = float(x0_data[x] + x2_data[x])
            w = float(x1_data[x] + x3_data[x])

            # EAST原始是旋转框，这里简化成水平框（适配方案A的[x, y, w, h]）
            end_x = int(offset_x + cos * x1_data[x] + sin * x2_data[x])
            end_y = int(offset_y - sin * x1_data[x] + cos * x2_data[x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            boxes.append([start_x, start_y, int(w), int(h)])
            confidences.append(score)

    return boxes, confidences


class EASTDetector:
    """
    EAST检测器封装：
    - 使用cv2.dnn加载.pb模型
    - 完成等比例缩放+Letterbox到固定输入尺寸
    - 解码特征图并做NMS
    - 输出与方案A兼容的水平检测框（还原到输入图像坐标）
    """

    def __init__(self) -> None:
        scheme_b_cfg = get_config("scheme_b")
        model_rel_path = scheme_b_cfg["east_model_path"]
        input_size = tuple(scheme_b_cfg.get("input_size", [320, 320]))

        # 记录配置
        self.input_w, self.input_h = int(input_size[0]), int(input_size[1])
        self.conf_threshold = float(scheme_b_cfg.get("conf_threshold", 0.5))
        self.nms_threshold = float(scheme_b_cfg.get("nms_threshold", 0.4))

        # 解析绝对路径并加载模型
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        model_path = os.path.join(project_root, model_rel_path)

        if not os.path.exists(model_path):
            raise DetectionError(f"EAST模型文件不存在，无法初始化：{model_path}")

        try:
            self.net = cv2.dnn.readNet(model_path)
        except Exception as e:
            raise DetectionError(f"EAST模型加载失败：{str(e)}")

        logger.info(
            f"EASTDetector初始化完成，模型：{model_path}，输入尺寸：{self.input_w}x{self.input_h}"
        )

    @handle_ocr_exception
    def detect(self, img: np.ndarray) -> List[List[int]]:
        """
        使用EAST进行文本检测

        Args:
            img: 输入BGR图像（已缩放后的图像）

        Returns:
            list: 检测框列表，每个元素为[x, y, w, h]，坐标基于输入图像尺寸
        """
        if img is None or img.size == 0:
            raise DetectionError("EAST检测收到空图像")

        orig_h, orig_w = img.shape[:2]

        # 1. 等比例缩放+Letterbox到固定尺寸
        scale = min(self.input_w / orig_w, self.input_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        padded = np.full((self.input_h, self.input_w, 3), 255, dtype=np.uint8)
        dx = (self.input_w - new_w) // 2
        dy = (self.input_h - new_h) // 2
        padded[dy : dy + new_h, dx : dx + new_w] = resized

        # 2. 构建blob并前向推理
        blob = cv2.dnn.blobFromImage(
            padded,
            1.0,
            (self.input_w, self.input_h),
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        scores, geometry = self.net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        # 3. 解码候选框
        boxes, confidences = _decode_east_scores(scores, geometry, self.conf_threshold)
        if not boxes:
            logger.info("EAST未检测到文本区域")
            return []

        # 4. NMS（OpenCV Python绑定不支持关键字参数，这里使用位置参数调用）
        indices = cv2.dnn.NMSBoxes(
            [[b[0], b[1], b[2], b[3]] for b in boxes],
            confidences,
            self.conf_threshold,
            self.nms_threshold,
        )

        final_boxes: List[List[int]] = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]

                # 还原到 padded 坐标对应的原图坐标
                x0 = max((x - dx) / scale, 0)
                y0 = max((y - dy) / scale, 0)
                x1 = min((x + w - dx) / scale, orig_w)
                y1 = min((y + h - dy) / scale, orig_h)

                final_boxes.append(
                    [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                )

        logger.info(f"EAST检测完成，最终检测框数：{len(final_boxes)}")
        return final_boxes


# 单例实例，供调度器复用
_east_detector_instance: EASTDetector | None = None


def get_east_detector() -> EASTDetector:
    global _east_detector_instance
    if _east_detector_instance is None:
        _east_detector_instance = EASTDetector()
    return _east_detector_instance

