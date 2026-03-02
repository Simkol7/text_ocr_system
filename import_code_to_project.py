import os
import sys
import shutil
import logging

# ===================== 配置项 =====================
PROJECT_ROOT = "C:/Users/29707/Desktop/text_ocr_system"
ALLOW_OVERWRITE_NON_EMPTY = False
# ===========================================================================

CODE_MAPPING = {
    # ===================== 1. 预处理模块（preprocess_module） =====================
    # 1.1 灰度化
    f"{PROJECT_ROOT}/algorithm/preprocess_module/gray_process.py": '''"""灰度化模块：将彩色图像转为灰度图，含有效性校验"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError

@handle_ocr_exception
def to_gray(img: np.ndarray) -> np.ndarray:
    """
    图像灰度化：彩色图转灰度图，灰度图直接返回

    Args:
        img: OpenCV格式图像（BGR）

    Returns:
        np.ndarray: 灰度图像（单通道）
    """
    if len(img.shape) == 2:
        logger.info("图像已是灰度图，无需处理")
        return img

    if len(img.shape) != 3 or img.shape[2] != 3:
        raise PreprocessError(f"图像通道数异常（{img.shape}），无法灰度化")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.info("图像灰度化完成，尺寸：{}".format(gray_img.shape))
    return gray_img''',

    # 1.2 CLAHE自适应直方图均衡化
    f"{PROJECT_ROOT}/algorithm/preprocess_module/clahe_enhance.py": '''"""CLAHE增强模块：解决光照不均问题，提升文本对比度"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def clahe_enhance(gray_img: np.ndarray) -> np.ndarray:
    """
    CLAHE自适应直方图均衡化：仅对灰度图生效

    Args:
        gray_img: 灰度图像（单通道）

    Returns:
        np.ndarray: CLAHE增强后的灰度图
    """
    if len(gray_img.shape) != 2:
        raise PreprocessError(f"CLAHE仅支持灰度图，当前图像维度：{gray_img.shape}")

    # 从配置读取CLAHE参数
    clahe_config = get_config("preprocess.clahe")
    clip_limit = clahe_config["clip_limit"]
    tile_grid_size = tuple(clahe_config["tile_grid_size"])

    # 初始化CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(gray_img)
    logger.info(f"CLAHE增强完成，参数：clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
    return clahe_img''',

    # 1.3 降噪模块
    f"{PROJECT_ROOT}/algorithm/preprocess_module/noise_removal.py": '''"""降噪模块：中值滤波去除椒盐噪点，可选高斯滤波"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def remove_noise(gray_img: np.ndarray, method: str = "median") -> np.ndarray:
    """
    图像降噪：默认中值滤波（适配文本图像）

    Args:
        gray_img: 灰度图像
        method: 降噪方法（median/ gaussian）

    Returns:
        np.ndarray: 降噪后的图像
    """
    if len(gray_img.shape) != 2:
        raise PreprocessError(f"降噪仅支持灰度图，当前维度：{gray_img.shape}")

    kernel_size = get_config("preprocess.noise_removal.kernel_size")
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保核尺寸为奇数
        logger.warning(f"降噪核尺寸需为奇数，已修正为{kernel_size}")

    if method == "median":
        denoised_img = cv2.medianBlur(gray_img, kernel_size)
    elif method == "gaussian":
        denoised_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    else:
        raise PreprocessError(f"不支持的降噪方法：{method}，仅支持median/gaussian")

    logger.info(f"图像降噪完成，方法：{method}，核尺寸：{kernel_size}")
    return denoised_img''',

    # 1.4 二值化模块
    f"{PROJECT_ROOT}/algorithm/preprocess_module/binarization.py": '''"""二值化模块：全局/自适应二值化，将灰度图转为黑白图"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def binarize(gray_img: np.ndarray) -> np.ndarray:
    """
    图像二值化：从配置读取二值化类型，自适应二值化优先

    Args:
        gray_img: 灰度图像

    Returns:
        np.ndarray: 二值化图像（黑白）
    """
    if len(gray_img.shape) != 2:
        raise PreprocessError(f"二值化仅支持灰度图，当前维度：{gray_img.shape}")

    bin_config = get_config("preprocess.binarization")
    bin_type = bin_config["type"]

    if bin_type == "global":
        # 全局二值化（OTSU自动阈值）
        _, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif bin_type == "adaptive":
        # 自适应二值化（适配局部光照不均）
        block_size = bin_config["adaptive_block_size"]
        c = bin_config["adaptive_c"]
        if block_size % 2 == 0:
            block_size += 1
            logger.warning(f"自适应二值化块尺寸需为奇数，已修正为{block_size}")
        bin_img = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c
        )
    else:
        raise PreprocessError(f"不支持的二值化类型：{bin_type}，仅支持global/adaptive")

    logger.info(f"图像二值化完成，类型：{bin_type}")
    return bin_img''',

    # 1.5 形态学优化
    f"{PROJECT_ROOT}/algorithm/preprocess_module/morphology_optim.py": '''"""形态学优化模块：膨胀/腐蚀/开运算/闭运算修复字符边缘"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def morphology_optimize(bin_img: np.ndarray) -> np.ndarray:
    """
    形态学优化：闭运算填充字符内部孔洞，开运算去除小噪点

    Args:
        bin_img: 二值化图像

    Returns:
        np.ndarray: 形态学优化后的二值图
    """
    if len(bin_img.shape) != 2:
        raise PreprocessError(f"形态学操作仅支持二值图，当前维度：{bin_img.shape}")

    kernel_size = get_config("preprocess.morphology.kernel_size")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # 先闭运算（填充孔洞），再开运算（去除噪点）
    close_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    morph_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)

    logger.info(f"形态学优化完成，核尺寸：{kernel_size}")
    return morph_img''',

    # 1.6 倾斜校正
    f"{PROJECT_ROOT}/algorithm/preprocess_module/orientation_fix.py": '''"""倾斜校正模块：基于最小外接矩形计算倾斜角度，自动校正"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, PreprocessError, get_config

@handle_ocr_exception
def fix_orientation(bin_img: np.ndarray, gray_img: np.ndarray) -> np.ndarray:
    """
    倾斜校正：基于二值图轮廓计算倾斜角度，对灰度图进行旋转校正

    Args:
        bin_img: 二值化图像（用于轮廓提取）
        gray_img: 灰度图像（需要校正的图像）

    Returns:
        np.ndarray: 校正后的灰度图
    """
    if len(bin_img.shape) != 2 or len(gray_img.shape) != 2:
        raise PreprocessError("倾斜校正仅支持二值图+灰度图输入")

    # 提取轮廓（仅提取最外层轮廓）
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("未检测到轮廓，跳过倾斜校正")
        return gray_img

    # 合并所有轮廓为一个大轮廓
    all_contours = np.vstack(contours[i] for i in range(len(contours)))
    # 计算最小外接矩形
    rect = cv2.minAreaRect(all_contours)
    angle = rect[-1]  # 倾斜角度

    # 角度校正（OpenCV计算的角度需要转换）
    max_angle = get_config("preprocess.orientation_fix.max_angle")
    angle_threshold = get_config("preprocess.orientation_fix.angle_threshold")

    if angle < -45:
        angle += 90
    if abs(angle) < angle_threshold or abs(angle) > max_angle:
        logger.info(f"倾斜角度{angle:.2f}°（阈值{angle_threshold}°），跳过校正")
        return gray_img

    # 旋转图像（保持尺寸，黑色填充）
    h, w = gray_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    fixed_img = cv2.warpAffine(
        gray_img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )

    logger.info(f"倾斜校正完成，校正角度：{angle:.2f}°")
    return fixed_img''',

    # 1.7 预处理流程整合
    f"{PROJECT_ROOT}/algorithm/preprocess_module/preprocess_core.py": '''"""预处理流程整合：按配置步骤自动执行所有预处理"""
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
    return current_img, process_log''',

    # ===================== 2. 检测模块（detection_module） =====================
    # 2.1 轮廓提取
    f"{PROJECT_ROOT}/algorithm/detection_module/contour_extract.py": '''"""轮廓提取模块：从二值图提取文本轮廓，含层级筛选"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError

@handle_ocr_exception
def extract_contours(bin_img: np.ndarray) -> list:
    """
    提取文本轮廓：仅提取外层轮廓，过滤极小轮廓

    Args:
        bin_img: 二值化图像（黑白）

    Returns:
        list: 轮廓列表（cv2轮廓格式）
    """
    if len(bin_img.shape) != 2:
        raise DetectionError(f"轮廓提取仅支持二值图，当前维度：{bin_img.shape}")

    # 提取轮廓（RETR_EXTERNAL：仅最外层轮廓）
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise DetectionError("未检测到任何轮廓")

    logger.info(f"轮廓提取完成，原始轮廓数：{len(contours)}")
    return contours''',

    # 2.2 轮廓筛选
    f"{PROJECT_ROOT}/algorithm/detection_module/contour_filter.py": '''"""轮廓筛选模块：按面积、长宽比、实心度过滤无效轮廓"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError, get_config

@handle_ocr_exception
def filter_contours(contours: list) -> list:
    """
    筛选有效文本轮廓：去除过小/过长/过扁/实心度低的轮廓

    Args:
        contours: 原始轮廓列表

    Returns:
        list: 筛选后的轮廓列表
    """
    if not contours:
        raise DetectionError("轮廓列表为空，无法筛选")

    filter_config = get_config("detection.contour_filter")
    min_area = filter_config["min_area"]
    aspect_ratio_min, aspect_ratio_max = filter_config["aspect_ratio_range"]
    solidity_threshold = filter_config["solidity_threshold"]
    min_height = filter_config["min_height"]

    filtered_contours = []
    for cnt in contours:
        # 1. 面积筛选
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 2. 外接矩形筛选（长宽比、高度）
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0
        if not (aspect_ratio_min <= aspect_ratio <= aspect_ratio_max) or h < min_height:
            continue

        # 3. 实心度筛选（轮廓面积/凸包面积）
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < solidity_threshold:
            continue

        filtered_contours.append(cnt)

    logger.info(f"轮廓筛选完成，筛选后轮廓数：{len(filtered_contours)}（原始：{len(contours)}）")
    return filtered_contours''',

    # 2.3 轮廓合并
    f"{PROJECT_ROOT}/algorithm/detection_module/contour_merge.py": '''"""轮廓合并模块：合并相邻/重叠的文本轮廓，生成检测框"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError, get_config

@handle_ocr_exception
def merge_contours(contours: list) -> list:
    """
    合并相邻轮廓：按水平距离合并，生成文本行检测框

    Args:
        contours: 筛选后的轮廓列表

    Returns:
        list: 合并后的检测框列表（每个框：[x, y, w, h]）
    """
    if not contours:
        raise DetectionError("轮廓列表为空，无法合并")

    merge_distance = get_config("detection.merge_distance")

    # 按轮廓y坐标排序（从上到下）
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

    boxes = []
    current_box = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if current_box is None:
            current_box = [x, y, x + w, y + h]
        else:
            # 判断是否在同一行（y轴重叠/距离近）
            if abs(y - current_box[1]) < merge_distance or abs((y + h) - current_box[3]) < merge_distance:
                # 合并框：取最小x、最小y、最大x、最大y
                current_box[0] = min(current_box[0], x)
                current_box[1] = min(current_box[1], y)
                current_box[2] = max(current_box[2], x + w)
                current_box[3] = max(current_box[3], y + h)
            else:
                # 保存当前框，新建框
                boxes.append([current_box[0], current_box[1], current_box[2] - current_box[0], current_box[3] - current_box[1]])
                current_box = [x, y, x + w, y + h]

    # 保存最后一个框
    if current_box is not None:
        boxes.append([current_box[0], current_box[1], current_box[2] - current_box[0], current_box[3] - current_box[1]])

    logger.info(f"轮廓合并完成，检测框数：{len(boxes)}")
    return boxes''',

    # 2.4 检测框坐标还原
    f"{PROJECT_ROOT}/algorithm/detection_module/box_restore.py": '''"""检测框坐标还原：适配预处理的缩放/旋转，还原到原始图像坐标"""
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError

@handle_ocr_exception
def restore_boxes(boxes: list, scale_factor: float = 1.0) -> list:
    """
    还原检测框坐标：将预处理后图像的框坐标还原到原始图像

    Args:
        boxes: 预处理后图像的检测框列表（[x, y, w, h]）
        scale_factor: 图像缩放系数（预处理时的缩放比例）

    Returns:
        list: 还原后的检测框列表
    """
    if not boxes:
        raise DetectionError("检测框列表为空，无法还原")
    if scale_factor <= 0:
        raise DetectionError(f"缩放系数非法：{scale_factor}")

    restored_boxes = []
    for box in boxes:
        x, y, w, h = box
        # 还原坐标（除以缩放系数，取整）
        x_restored = int(x / scale_factor)
        y_restored = int(y / scale_factor)
        w_restored = int(w / scale_factor)
        h_restored = int(h / scale_factor)
        restored_boxes.append([x_restored, y_restored, w_restored, h_restored])

    logger.info(f"检测框坐标还原完成，缩放系数：{scale_factor}")
    return restored_boxes''',

    # 2.5 检测流程整合
    f"{PROJECT_ROOT}/algorithm/detection_module/detection_core.py": '''"""检测流程整合：从二值图到最终检测框，含可视化标注"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, DetectionError
from .contour_extract import extract_contours
from .contour_filter import filter_contours
from .contour_merge import merge_contours
from .box_restore import restore_boxes

@handle_ocr_exception
def run_detection(bin_img: np.ndarray, scale_factor: float = 1.0) -> tuple[list, np.ndarray]:
    """
    执行完整文本检测流程，返回检测框+标注图像

    Args:
        bin_img: 二值化图像
        scale_factor: 图像缩放系数（用于还原坐标）

    Returns:
        tuple: (还原后的检测框列表, 标注检测框的图像)
    """
    logger.info("开始文本检测流程")

    # 1. 轮廓提取
    contours = extract_contours(bin_img)
    # 2. 轮廓筛选
    filtered_contours = filter_contours(contours)
    if not filtered_contours:
        raise DetectionError("筛选后无有效轮廓，检测失败")
    # 3. 轮廓合并（生成检测框）
    boxes = merge_contours(filtered_contours)
    # 4. 坐标还原
    restored_boxes = restore_boxes(boxes, scale_factor)

    # 5. 可视化标注（在二值图上绘制检测框）
    vis_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    logger.info(f"检测流程完成，最终检测框数：{len(restored_boxes)}")
    return restored_boxes, vis_img''',

    # ===================== 3. 识别模块（recognition_module） =====================
    # 3.1 ROI裁剪
    f"{PROJECT_ROOT}/algorithm/recognition_module/roi_crop.py": '''"""ROI裁剪模块：按检测框裁剪文本区域，含有效性校验"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, RecognitionError

@handle_ocr_exception
def crop_roi(img: np.ndarray, box: list) -> np.ndarray:
    """
    裁剪文本ROI区域：检测框坐标校验+边界修正

    Args:
        img: 预处理后的灰度图
        box: 检测框（[x, y, w, h]）

    Returns:
        np.ndarray: 裁剪后的ROI图像
    """
    if len(img.shape) != 2:
        raise RecognitionError(f"ROI裁剪仅支持灰度图，当前维度：{img.shape}")

    x, y, w, h = box
    h_img, w_img = img.shape

    # 边界校验（防止越界）
    x = max(0, x)
    y = max(0, y)
    x2 = min(x + w, w_img)
    y2 = min(y + h, h_img)

    if x2 <= x or y2 <= y:
        raise RecognitionError(f"检测框越界，无法裁剪：{box}（图像尺寸：{w_img}x{h_img}）")

    roi_img = img[y:y2, x:x2]
    logger.info(f"ROI裁剪完成，检测框：{box}，ROI尺寸：{roi_img.shape}")
    return roi_img''',

    # 3.2 ROI预处理
    f"{PROJECT_ROOT}/algorithm/recognition_module/roi_optim.py": '''"""ROI预处理模块：精细化优化裁剪后的文本区域"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, RecognitionError

@handle_ocr_exception
def optimize_roi(roi_img: np.ndarray) -> np.ndarray:
    """
    ROI精细化优化：提升小文本区域的识别率

    Args:
        roi_img: 裁剪后的ROI灰度图

    Returns:
        np.ndarray: 优化后的ROI图像
    """
    if len(roi_img.shape) != 2:
        raise RecognitionError(f"ROI优化仅支持灰度图，当前维度：{roi_img.shape}")

    # 1. 自适应阈值二值化（针对小区域）
    h, w = roi_img.shape
    block_size = min(15, max(3, h // 2, w // 2))
    if block_size % 2 == 0:
        block_size += 1
    roi_bin = cv2.adaptiveThreshold(
        roi_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 1
    )

    # 2. 轻微膨胀（增强字符边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    roi_optim = cv2.dilate(roi_bin, kernel, iterations=1)

    logger.info(f"ROI优化完成，块尺寸：{block_size}")
    return roi_optim''',

    # 3.3 Tesseract调用封装
    f"{PROJECT_ROOT}/algorithm/recognition_module/tesseract_call.py": '''"""Tesseract调用模块：动态配置PSM/语言，适配不同文本类型"""
import cv2
import numpy as np
import pytesseract
from algorithm.core import logger, handle_ocr_exception, RecognitionError, get_tesseract_path, get_config

# 设置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()

@handle_ocr_exception
def call_tesseract(roi_img: np.ndarray, psm_type: str = "single_line") -> tuple[str, float]:
    """
    调用Tesseract识别文本，返回识别结果+置信度

    Args:
        roi_img: 优化后的ROI图像
        psm_type: PSM模式类型（single_line/paragraph/single_char）

    Returns:
        tuple: (识别文本, 平均置信度)
    """
    if len(roi_img.shape) != 2:
        raise RecognitionError(f"Tesseract仅支持灰度/二值图，当前维度：{roi_img.shape}")

    # 获取PSM配置
    psm_mapping = get_config("recognition.psm_mapping")
    lang = get_config("recognition.lang")

    if psm_type not in psm_mapping:
        psm_type = "single_line"
        logger.warning(f"未知PSM类型：{psm_type}，使用默认值single_line")

    psm = psm_mapping[psm_type]
    config = f"--psm {psm} -l {lang}"

    # 调用Tesseract（获取详细结果含置信度）
    try:
        details = pytesseract.image_to_data(
            roi_img, config=config, output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        raise RecognitionError(f"Tesseract调用失败：{str(e)}")

    # 提取有效文本和置信度
    text = ""
    confidences = []
    for i, conf in enumerate(details["conf"]):
        if conf != -1:  # -1表示无文本
            text += details["text"][i] + " "
            confidences.append(conf)

    text = text.strip()
    avg_conf = np.mean(confidences) if confidences else 0.0

    logger.info(f"Tesseract识别完成，PSM：{psm}，文本：{text}，置信度：{avg_conf:.2f}")
    return text, avg_conf''',

    # 3.4 识别流程整合
    f"{PROJECT_ROOT}/algorithm/recognition_module/recognition_core.py": '''"""识别流程整合：关联检测框+ROI裁剪+识别，输出带坐标的文本结果"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, RecognitionError
from .roi_crop import crop_roi
from .roi_optim import optimize_roi
from .tesseract_call import call_tesseract

@handle_ocr_exception
def run_recognition(img: np.ndarray, boxes: list) -> list:
    """
    执行完整文本识别流程，返回带坐标的识别结果

    Args:
        img: 预处理后的灰度图
        boxes: 还原后的检测框列表（[x, y, w, h]）

    Returns:
        list: 识别结果列表（每个元素：{"box": [x,y,w,h], "text": "", "confidence": 0.0}）
    """
    if not boxes:
        raise RecognitionError("检测框列表为空，无法识别")
    if len(img.shape) != 2:
        raise RecognitionError(f"识别仅支持灰度图，当前维度：{img.shape}")

    logger.info(f"开始文本识别流程，检测框数：{len(boxes)}")
    results = []

    for box in boxes:
        try:
            # 1. ROI裁剪
            roi_img = crop_roi(img, box)
            # 2. ROI优化
            roi_optim = optimize_roi(roi_img)
            # 3. 调用Tesseract
            text, confidence = call_tesseract(roi_optim)

            # 保存结果
            results.append({
                "box": box,
                "text": text,
                "confidence": round(confidence, 2)
            })
        except Exception as e:
            logger.error(f"检测框{box}识别失败：{str(e)}")
            results.append({
                "box": box,
                "text": "",
                "confidence": 0.0
            })

    logger.info(f"识别流程完成，有效识别文本数：{len([r for r in results if r['text']])}")
    return results''',

    # ===================== 4. 输出模块（output_module） =====================
    # 4.1 结果标准化
    f"{PROJECT_ROOT}/algorithm/output_module/result_format.py": '''"""结果标准化模块：将识别结果转为JSON格式，统一输出规范"""
import json
import time
from algorithm.core import logger, handle_ocr_exception

@handle_ocr_exception
def format_result(original_path: str, boxes: list, recognition_results: list) -> dict:
    """
    标准化识别结果：包含元信息+检测框+识别文本+置信度

    Args:
        original_path: 原始图像路径
        boxes: 检测框列表
        recognition_results: 识别结果列表

    Returns:
        dict: 标准化结果（可直接转为JSON）
    """
    # 基础元信息
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "original_image": original_path,
        "total_boxes": len(boxes),
        "valid_recognitions": len([r for r in recognition_results if r["text"]]),
        "results": []
    }

    # 关联检测框和识别结果
    for i, (box, recog) in enumerate(zip(boxes, recognition_results)):
        result["results"].append({
            "index": i + 1,
            "bounding_box": {
                "x": box[0],
                "y": box[1],
                "width": box[2],
                "height": box[3]
            },
            "text": recog["text"],
            "confidence": recog["confidence"]
        })

    logger.info("识别结果标准化完成")
    return result''',

    # 4.2 结果保存
    f"{PROJECT_ROOT}/algorithm/output_module/result_save.py": '''"""结果保存模块：保存识别文本到TXT/JSON，保存标注图像"""
import os
import json
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception, get_config

@handle_ocr_exception
def save_result(formatted_result: dict, vis_img: np.ndarray, original_path: str) -> None:
    """
    保存识别结果：
    - JSON/TXT：保存文本结果
    - 图像：保存标注检测框和文本的图像

    Args:
        formatted_result: 标准化结果
        vis_img: 标注检测框的图像
        original_path: 原始图像路径
    """
    save_format = get_config("output.save_format")
    save_path = get_config("output.image_save_path")

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 提取文件名（无后缀）
    filename = os.path.splitext(os.path.basename(original_path))[0]

    # 1. 保存文本结果
    if save_format == "json":
        json_path = os.path.join(save_path, f"{filename}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(formatted_result, f, ensure_ascii=False, indent=4)
        logger.info(f"JSON结果保存到：{json_path}")
    else:
        txt_path = os.path.join(save_path, f"{filename}_result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"识别时间：{formatted_result['timestamp']}\\n")
            f.write(f"原始图像：{formatted_result['original_image']}\\n")
            f.write(f"总检测框数：{formatted_result['total_boxes']}\\n")
            f.write(f"有效识别数：{formatted_result['valid_recognitions']}\\n\\n")
            for res in formatted_result["results"]:
                f.write(f"[{res['index']}] 坐标({res['bounding_box']['x']},{res['bounding_box']['y']})：{res['text']}（置信度：{res['confidence']}）\\n")
        logger.info(f"TXT结果保存到：{txt_path}")

    # 2. 保存标注图像
    img_save_path = os.path.join(save_path, f"{filename}_annotated.jpg")
    cv2.imwrite(img_save_path, vis_img)
    logger.info(f"标注图像保存到：{img_save_path}")''',

    # 4.3 结果展示
    f"{PROJECT_ROOT}/algorithm/output_module/result_show.py": '''"""结果展示模块：控制台格式化打印+PyQt5界面展示"""
import cv2
import numpy as np
from algorithm.core import logger, handle_ocr_exception

@handle_ocr_exception
def print_result(formatted_result: dict) -> None:
    """
    控制台格式化打印识别结果

    Args:
        formatted_result: 标准化结果
    """
    logger.info("\\n===== 识别结果汇总 =====")
    print(f"识别时间：{formatted_result['timestamp']}")
    print(f"原始图像：{formatted_result['original_image']}")
    print(f"总检测框数：{formatted_result['total_boxes']}")
    print(f"有效识别数：{formatted_result['valid_recognitions']}")
    print("\\n===== 详细识别结果 =====")
    for res in formatted_result["results"]:
        if res["text"]:
            print(f"[{res['index']}] 坐标({res['bounding_box']['x']},{res['bounding_box']['y']})：{res['text']}（置信度：{res['confidence']}）")
        else:
            print(f"[{res['index']}] 坐标({res['bounding_box']['x']},{res['bounding_box']['y']})：无有效文本")

@handle_ocr_exception
def show_image(vis_img: np.ndarray, window_name: str = "OCR Result") -> None:
    """
    可视化展示标注图像（调试用）

    Args:
        vis_img: 标注后的图像
        window_name: 窗口名称
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()''',

    # ===================== 5. 输入模块补充（batch_loader.py） =====================
    f"{PROJECT_ROOT}/algorithm/input_module/batch_loader.py": '''"""批量图像加载模块：支持文件夹批量读取，含进度条/异常跳过"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from algorithm.core import logger, handle_ocr_exception, InputError, get_config
from .image_loader import load_image

@handle_ocr_exception
def load_batch_images(folder_path: str, scale_factor: float = 1.0) -> dict:
    """
    批量加载文件夹内的所有图像，跳过无效文件

    Args:
        folder_path: 图像文件夹路径
        scale_factor: 缩放系数

    Returns:
        dict: 加载结果（key：图像路径，value：(图像, 缩放系数, 是否成功)）
    """
    if not os.path.exists(folder_path):
        raise InputError(f"文件夹路径不存在：{folder_path}")
    if not os.path.isdir(folder_path):
        raise InputError(f"路径不是文件夹：{folder_path}")

    valid_formats = get_config("input.valid_formats")
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[-1].lower().lstrip(".") in valid_formats
    ]

    if not image_files:
        raise InputError(f"文件夹{folder_path}内无有效图像（支持格式：{','.join(valid_formats)}）")

    logger.info(f"开始批量加载图像，共{len(image_files)}个文件")
    batch_results = {}

    # 进度条展示
    for img_path in tqdm(image_files, desc="Loading Images"):
        try:
            img, scale, success = load_image(img_path, scale_factor)
            batch_results[img_path] = (img, scale, success)
        except Exception as e:
            logger.error(f"加载图像{img_path}失败：{str(e)}，跳过")
            batch_results[img_path] = (None, scale_factor, False)

    # 统计加载结果
    success_count = len([v for v in batch_results.values() if v[2]])
    logger.info(f"批量加载完成，成功：{success_count}/{len(image_files)}")
    return batch_results''',

    # ===================== 6. 测试模块核心文件 =====================
    # 6.1 测试数据集管理
    f"{PROJECT_ROOT}/test/test_cases/test_data.py": '''"""测试数据集管理：标注数据加载、测试集分类"""
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

    # 检查测试集路径
    for name, path in test_sets.items():
        if not os.path.exists(path):
            logger.warning(f"测试集{name}路径不存在：{path}，已跳过")
            del test_sets[name]

    logger.info(f"加载测试集完成，可用测试集：{list(test_sets.keys())}")
    return test_sets''',

    # 6.2 单模块测试
    f"{PROJECT_ROOT}/test/test_scripts/module_test.py": '''"""单模块测试：单独验证预处理/检测/识别模块"""
import os
import cv2
import numpy as np
from algorithm.core import logger, config_manager
from algorithm.input_module.image_loader import load_image
from algorithm.preprocess_module.preprocess_core import run_preprocess
from algorithm.detection_module.detection_core import run_detection
from algorithm.recognition_module.recognition_core import run_recognition

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
        logger.error("测试图像不存在，请先放入test/test_cases/test_images/目录")''',

    # 6.3 全链路测试
    f"{PROJECT_ROOT}/test/test_scripts/full_test.py": '''"""全链路测试：单张图像从输入到输出的完整OCR流程"""
import os
import cv2
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

    # 6. 结果保存/打印
    if save_result_flag:
        save_result(formatted_result, vis_img, img_path)
    print_result(formatted_result)

    logger.info("===== 全链路OCR测试完成 =====")
    return formatted_result

if __name__ == "__main__":
    # 替换为你的测试图像路径
    test_img_path = "./test/test_cases/test_images/test.jpg"
    if os.path.exists(test_img_path):
        full_ocr_test(test_img_path)
    else:
        logger.error("测试图像不存在，请先放入test/test_cases/test_images/目录")''',

    # 6.4 批量测试
    f"{PROJECT_ROOT}/test/test_scripts/batch_test.py": '''"""批量测试：文件夹内所有图像测试，计算识别准确率/召回率"""
import os
import numpy as np
from algorithm.core import logger
from algorithm.input_module.batch_loader import load_batch_images
from algorithm.preprocess_module.preprocess_core import run_preprocess
from algorithm.detection_module.detection_core import run_detection
from algorithm.recognition_module.recognition_core import run_recognition
from algorithm.output_module.result_format import format_result
from test.test_cases.test_data import load_annotation

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

        # 单图像OCR流程
        processed_img, process_log = run_preprocess(img)
        bin_img = process_log["bin_img"]
        boxes, _ = run_detection(bin_img, scale_factor)
        recognition_results = run_recognition(processed_img, boxes)
        formatted_result = format_result(img_path, boxes, recognition_results)
        batch_formatted[img_path] = formatted_result

        # 计算指标（如果有标注）
        if img_path in annotation:
            predicted = [r["text"] for r in formatted_result["results"]]
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
    batch_ocr_test(test_folder, annotation_path)''',

    # 6.5 优化效果对比测试
    f"{PROJECT_ROOT}/test/test_scripts/optimization_test.py": '''"""优化效果对比测试：测试CLAHE/倾斜校正等优化点的提升效果"""
import os
import cv2
import numpy as np
from algorithm.core import logger, config_manager
from algorithm.input_module.image_loader import load_image
from algorithm.preprocess_module.preprocess_core import run_preprocess
from algorithm.detection_module.detection_core import run_detection
from algorithm.recognition_module.recognition_core import run_recognition
from test.test_scripts.batch_test import calculate_metrics
from test.test_cases.test_data import load_annotation

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
        logger.error("测试图像不存在，请先放入test/test_cases/test_images/challenge/目录")''',

    # ===================== 已导入的核心文件（重复以确保覆盖） =====================
    f"{PROJECT_ROOT}/algorithm/core/logger.py": '''import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name: str = "text_ocr_system") -> logging.Logger:
    """
    配置滚动日志器：单个文件最大10MB，最多保留5个备份，避免日志体积爆炸
    适配批量测试场景，符合毕设工程化规范
    """
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "ocr_system.log")
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.info(f"滚动日志器初始化完成，日志文件：{log_file}，单个文件最大10MB，保留5个备份")
    return logger

logger = setup_logger()''',

    f"{PROJECT_ROOT}/algorithm/core/config_manager.py": '''import os
import json
import logging
from typing import Dict, Optional, Any
from dotenv import load_dotenv

try:
    from PyQt5.QtWidgets import QFileDialog, QApplication
    import sys
    PYQT5_AVAILABLE = True
except ImportError:
    import tkinter as tk
    from tkinter import filedialog
    PYQT5_AVAILABLE = False

from .logger import logger

class ConfigManager:
    def __init__(self, config_path: str = "./config/params.json"):
        load_dotenv()
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.tesseract_path: Optional[str] = None

        self._load_config_file()
        self._get_tesseract_path()
        self._validate_config()

    def _load_config_file(self) -> None:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            logger.info(f"配置文件加载成功：{self.config_path}")
        except FileNotFoundError:
            logger.error(f"配置文件不存在：{self.config_path}")
            raise FileNotFoundError(f"请检查配置文件路径是否正确，当前路径：{self.config_path}")
        except json.JSONDecodeError:
            logger.error(f"配置文件格式错误（非合法JSON）：{self.config_path}")
            raise ValueError(f"配置文件{self.config_path}格式错误，请检查JSON语法")

    def _get_tesseract_path(self) -> None:
        self.tesseract_path = os.getenv("TESSERACT_PATH")
        if self.tesseract_path and os.path.exists(self.tesseract_path):
            logger.info(f"从环境变量加载Tesseract路径：{self.tesseract_path}")
            return

        logger.warning("未从环境变量找到有效Tesseract路径，将弹出选择框")
        if PYQT5_AVAILABLE:
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            selected_path, _ = QFileDialog.getOpenFileName(
                None,
                "选择Tesseract-OCR的tesseract.exe文件",
                "C:/Program Files/Tesseract-OCR/",
                "可执行文件 (*.exe);;所有文件 (*.*)"
            )
            app.processEvents()
        else:
            logger.warning("PyQt5未安装，临时使用tkinter选择文件（建议安装PyQt5：pip install PyQt5）")
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected_path = filedialog.askopenfilename(
                title="选择Tesseract-OCR的tesseract.exe文件",
                filetypes=[("可执行文件", "*.exe"), ("所有文件", "*.*")],
                initialdir="C:/Program Files/Tesseract-OCR/"
            )

        if selected_path and os.path.exists(selected_path):
            self.tesseract_path = selected_path
            logger.info(f"用户选择Tesseract路径：{self.tesseract_path}")
        else:
            logger.error("未选择有效Tesseract路径，OCR功能将无法使用")
            raise FileNotFoundError("未配置Tesseract路径，请选择正确的tesseract.exe文件")

    def _validate_config(self) -> None:
        clahe_clip_limit = self.config["preprocess"]["clahe"]["clip_limit"]
        if clahe_clip_limit <= 0:
            self.config["preprocess"]["clahe"]["clip_limit"] = 2.0
            logger.warning(f"CLAHE clip_limit非法（{clahe_clip_limit}），已修正为2.0")

        max_angle = self.config["preprocess"]["orientation_fix"]["max_angle"]
        if not (0 < max_angle <= 30):
            self.config["preprocess"]["orientation_fix"]["max_angle"] = 15
            logger.warning(f"倾斜校正max_angle非法（{max_angle}），已修正为15")

        psm_mapping = self.config["recognition"]["psm_mapping"]
        valid_psm_range = range(0, 14)
        default_psm = {"single_line": 7, "paragraph": 3, "single_char": 10}
        for key, value in psm_mapping.items():
            if value not in valid_psm_range:
                self.config["recognition"]["psm_mapping"][key] = default_psm[key]
                logger.warning(
                    f"PSM模式{key}={value}非法（需0-13），已修正为默认值{default_psm[key]}"
                )

        logger.info("配置参数校验完成，非法参数已自动修正")

    def get_config(self, key: Optional[str] = None) -> Any:
        if not key:
            return self.config
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            logger.error(f"配置键不存在：{key}")
            raise KeyError(f"配置文件中未找到键：{key}")

    def get_tesseract_path(self) -> str:
        if not self.tesseract_path:
            raise RuntimeError("Tesseract路径未配置")
        return self.tesseract_path

config_manager = ConfigManager()
get_config = config_manager.get_config
get_tesseract_path = config_manager.get_tesseract_path'''
}


def backup_file(file_path: str) -> None:
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    logger = logging.getLogger("import_code")
    logger.info(f"已备份非空文件：{file_path} → {backup_path}")


def write_code_to_files():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("import_code")

    if not os.path.exists(PROJECT_ROOT):
        logger.error(f"项目根目录不存在：{PROJECT_ROOT}，请检查路径配置")
        sys.exit(1)

    logger.info(f"===== 开始一键导入代码到项目：{PROJECT_ROOT} =====")
    success_count = 0
    fail_count = 0

    for file_path, code_content in CODE_MAPPING.items():
        try:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在：{file_path}，请先创建项目目录结构")
                fail_count += 1
                continue

            if os.path.getsize(file_path) > 0 and not ALLOW_OVERWRITE_NON_EMPTY:
                backup_file(file_path)
                logger.warning(f"文件非空且禁止覆盖：{file_path}，已备份，跳过写入")
                fail_count += 1
                continue

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            logger.info(f"✅ 代码写入成功：{file_path}")
            success_count += 1

        except Exception as e:
            logger.error(f"❌ 代码写入失败：{file_path}，异常：{str(e)}")
            fail_count += 1

    logger.info("\\n===== 一键导入完成 ======")
    logger.info(f"成功写入：{success_count} 个文件")
    logger.info(f"失败/跳过：{fail_count} 个文件")
    if fail_count > 0:
        logger.warning("部分文件写入失败，请检查日志后手动处理")


if __name__ == "__main__":
    print("⚠️  运行前请确保项目目录下的空文件未被修改，避免数据丢失！")
    confirm = input("是否继续？(y/n)：")
    if confirm.lower() == "y":
        write_code_to_files()
    else:
        print("操作已取消")