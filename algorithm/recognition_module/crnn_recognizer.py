"""CRNN文本识别模块：使用cv2.dnn加载ONNX模型，完成ROI归一化与CTC解码"""
import os
from typing import List

import cv2
import numpy as np

from algorithm.core import logger, handle_ocr_exception, RecognitionError, get_config


def _load_charset(keys_path: str) -> List[str]:
    """
    从keys.txt加载字符表，按行读取
    """
    charset: List[str] = []
    try:
        with open(keys_path, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.rstrip("\n")
                if ch:
                    charset.append(ch)
    except FileNotFoundError:
        raise RecognitionError(f"CRNN字典文件不存在：{keys_path}")

    if not charset:
        raise RecognitionError(f"CRNN字典文件为空：{keys_path}")

    logger.info(f"CRNN字典加载完成，共{len(charset)}个字符")
    return charset


def _ctc_decode(preds: np.ndarray, charset: List[str]) -> str:
    """
    简单CTC贪心解码：
    - preds: [seq_len, num_classes]
    - charset: 字符表，最后一个索引视为blank
    """
    seq_len, num_classes = preds.shape
    blank_index = num_classes - 1

    last_idx = blank_index
    text_chars: List[str] = []

    for t in range(seq_len):
        idx = int(np.argmax(preds[t]))
        if idx != blank_index and idx != last_idx:
            if 0 <= idx < len(charset):
                text_chars.append(charset[idx])
        last_idx = idx

    return "".join(text_chars)


class CRNNRecognizer:
    """
    CRNN识别器封装：
    - 使用cv2.dnn加载ONNX模型
    - 将ROI缩放到32x100，做简单归一化
    - 使用CTC贪心解码输出中英混合字符串
    """

    def __init__(self) -> None:
        scheme_b_cfg = get_config("scheme_b")
        model_rel_path = scheme_b_cfg["crnn_model_path"]
        keys_rel_path = scheme_b_cfg["keys_path"]

        # 解析绝对路径
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        model_path = os.path.join(project_root, model_rel_path)
        keys_path = os.path.join(project_root, keys_rel_path)

        if not os.path.exists(model_path):
            raise RecognitionError(f"CRNN模型文件不存在，无法初始化：{model_path}")

        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
        except Exception as e:
            raise RecognitionError(f"CRNN模型加载失败：{str(e)}")

        self.charset = _load_charset(keys_path)

        logger.info(
            f"CRNNRecognizer初始化完成，模型：{model_path}，字典：{keys_path}"
        )

    @handle_ocr_exception
    def recognize(self, roi_img: np.ndarray) -> tuple[str, float]:
        """
        使用CRNN对单个ROI进行识别

        Args:
            roi_img: BGR或灰度图像

        Returns:
            tuple: (识别文本, 置信度占位值)
        """
        if roi_img is None or roi_img.size == 0:
            raise RecognitionError("CRNN识别收到空ROI图像")

        # 1. 转灰度并缩放到32x100
        if len(roi_img.shape) == 3:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_img

        resized = cv2.resize(gray, (100, 32))
        norm = resized.astype("float32") / 255.0
        norm = (norm - 0.5) / 0.5  # 简单标准化到[-1,1]

        # 2. 构造输入张量：[1, 1, 32, 100]
        blob = norm[np.newaxis, np.newaxis, :, :]
        self.net.setInput(blob)
        preds = self.net.forward()

        if preds.ndim != 3:
            raise RecognitionError(f"CRNN输出维度异常：{preds.shape}")

        # 常见两种输出形状：
        # 1) [1, num_classes, seq_len]
        # 2) [1, seq_len, num_classes]
        p = preds[0]
        if p.shape[0] < p.shape[1]:
            # 形如[num_classes, seq_len] -> [seq_len, num_classes]
            preds_seq = p.transpose(1, 0)
        else:
            # 形如[seq_len, num_classes]
            preds_seq = p

        text = _ctc_decode(preds_seq, self.charset)
        # 暂时用占位置信度：平均最大概率
        max_probs = np.max(preds_seq, axis=1)
        confidence = float(np.mean(max_probs)) if max_probs.size > 0 else 0.0

        logger.info(f"CRNN识别完成，文本：{text}，置信度（估计）：{confidence:.4f}")
        return text, confidence


_crnn_instance: CRNNRecognizer | None = None


def get_crnn_recognizer() -> CRNNRecognizer:
    global _crnn_instance
    if _crnn_instance is None:
        _crnn_instance = CRNNRecognizer()
    return _crnn_instance

