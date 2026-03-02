"""结果保存模块：将识别结果保存为TXT/JSON，标注图像保存为JPG"""
import os
import json
import cv2
import numpy as np
from datetime import datetime
from algorithm.core import logger, get_config, handle_ocr_exception, OutputError


@handle_ocr_exception
def save_result(formatted_result: dict, vis_img: np.ndarray, img_path: str, project_root: str) -> None:
    """
    保存识别结果（TXT/JSON）和标注图像

    Args:
        formatted_result: 标准化识别结果
        vis_img: 标注了检测框的可视化图像
        img_path: 原始图像路径
        project_root: 项目根目录（从外部传递，避免计算错误）

    Raises:
        OutputError: 保存失败时抛出异常
    """
    # ===================== 核心修复：直接使用传递的项目根目录 =====================
    # 从配置读取保存格式（默认txt）
    save_format = get_config("output.save_format").lower()
    # 拼接保存目录的绝对路径（项目根目录下的test/test_results/）
    save_dir = os.path.join(project_root, "test", "test_results")

    # 强制创建保存目录（确保存在）
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"结果保存目录（确保存在）：{save_dir}")

    # 提取原始图像文件名（用于生成结果文件名）
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===================== 保存文本结果 =====================
    result_filename = f"{img_name}_result_{timestamp}.txt" if save_format == "txt" else f"{img_name}_result_{timestamp}.json"
    result_save_path = os.path.join(save_dir, result_filename)

    try:
        if save_format == "txt":
            # 生成TXT格式结果
            txt_content = f"识别时间：{formatted_result['timestamp']}\n"
            txt_content += f"原始图像：{formatted_result['img_path']}\n"
            txt_content += f"总检测框数：{formatted_result['total_boxes']}\n"
            txt_content += f"有效识别数：{formatted_result['valid_recognitions']}\n\n"
            txt_content += "===== 详细识别结果 =====\n"
            for idx, res in enumerate(formatted_result['recognitions'], 1):
                txt_content += f"[{idx}] 坐标({res['x']},{res['y']})：{res['text']}（置信度：{res['confidence']:.1f}）\n"

            with open(result_save_path, "w", encoding="utf-8") as f:
                f.write(txt_content)
        else:
            # 生成JSON格式结果
            with open(result_save_path, "w", encoding="utf-8") as f:
                json.dump(formatted_result, f, ensure_ascii=False, indent=2)

        logger.info(f"{save_format.upper()}结果保存到：{result_save_path}")
        # 新增：打印文件是否真的创建成功
        if os.path.exists(result_save_path):
            logger.info(f"✅ TXT文件创建成功，大小：{os.path.getsize(result_save_path)} 字节")
        else:
            logger.error(f"❌ TXT文件创建失败，路径：{result_save_path}")
    except Exception as e:
        raise OutputError(f"保存{save_format}结果失败：{str(e)}")

    # ===================== 保存标注图像 =====================
    img_filename = f"{img_name}_annotated_{timestamp}.jpg"
    img_save_path = os.path.join(save_dir, img_filename)

    try:
        # 确保图像是BGR格式（OpenCV默认），避免保存错误
        if len(vis_img.shape) == 2:  # 灰度图转BGR
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        # 强制保存，指定压缩参数
        cv2.imwrite(img_save_path, vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        logger.info(f"标注图像保存到：{img_save_path}")
        # 新增：打印图像是否真的创建成功
        if os.path.exists(img_save_path):
            logger.info(f"✅ 图像文件创建成功，大小：{os.path.getsize(img_save_path)} 字节")
        else:
            logger.error(f"❌ 图像文件创建失败，路径：{img_save_path}")
    except Exception as e:
        raise OutputError(f"保存标注图像失败：{str(e)}")