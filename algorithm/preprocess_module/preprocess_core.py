"""预处理流程整合：按配置步骤自动执行所有预处理"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config
from .gray_process import to_gray
from .clahe_enhance import clahe_enhance
from .noise_removal import remove_noise
from .binarization import binarize
from .morphology_optim import morphology_optimize
from .orientation_fix import fix_orientation

@handle_ocr_exception
def run_preprocess(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    执行完整预处理流程，返回处理后图像+步骤日志

    Args:
        img: 原始BGR图像

    Returns:
        tuple: (处理后的灰度图, 预处理步骤日志)
    """
    process_log = {}
    steps = get_config("preprocess.steps") if "preprocess.steps" in get_config() else [
        "gray", "clahe", "noise_removal", "binarization", "morphology", "orientation_fix"
    ]

    logger.info(f"开始预处理流程，步骤：{steps}")
    current_img = img.copy()

    # 按步骤执行预处理
    for step in steps:
        try:
            if step == "gray":
                current_img = to_gray(current_img)
            elif step == "clahe":
                current_img = clahe_enhance(current_img)
            elif step == "noise_removal":
                current_img = remove_noise(current_img)
            elif step == "binarization":
                bin_img = binarize(current_img)  # 二值图单独保存，用于倾斜校正
                process_log["bin_img"] = bin_img
            elif step == "morphology":
                process_log["bin_img"] = morphology_optimize(process_log["bin_img"])
            elif step == "orientation_fix":
                current_img = fix_orientation(process_log["bin_img"], current_img)
            else:
                logger.warning(f"未知预处理步骤：{step}，跳过")
                continue

            process_log[step] = "success"
            logger.info(f"预处理步骤{step}执行完成")
        except Exception as e:
            process_log[step] = f"failed: {str(e)}"
            logger.error(f"预处理步骤{step}执行失败：{str(e)}")
            # 非关键步骤失败，继续执行后续步骤
            if step not in ["gray", "binarization"]:
                continue
            else:
                raise PreprocessError(f"核心预处理步骤{step}失败：{str(e)}")

    logger.info("预处理流程执行完成")
    return current_img, process_log