import os
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
    def __init__(self, config_path: str = None):
        load_dotenv()

        # 核心修复：基于脚本位置计算项目根目录和配置文件绝对路径
        if config_path is None:
            # 当前脚本路径：algorithm/core/config_manager.py
            current_script_path = os.path.abspath(__file__)
            # 向上回溯3级，得到项目根目录：text_ocr_system/
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
            # 拼接配置文件绝对路径
            config_path = os.path.join(project_root, "config", "params.json")

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
        # 确保preprocess配置存在
        if "preprocess" not in self.config:
            self.config["preprocess"] = {
                "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
                "orientation_fix": {"max_angle": 15, "angle_threshold": 0.5}
            }

        clahe_clip_limit = self.config["preprocess"]["clahe"]["clip_limit"]
        if clahe_clip_limit <= 0:
            self.config["preprocess"]["clahe"]["clip_limit"] = 2.0
            logger.warning(f"CLAHE clip_limit非法（{clahe_clip_limit}），已修正为2.0")

        max_angle = self.config["preprocess"]["orientation_fix"]["max_angle"]
        if not (0 < max_angle <= 30):
            self.config["preprocess"]["orientation_fix"]["max_angle"] = 15
            logger.warning(f"倾斜校正max_angle非法（{max_angle}），已修正为15")

        # 确保recognition配置存在
        if "recognition" not in self.config:
            self.config["recognition"] = {"psm_mapping": {"single_line": 7, "paragraph": 3, "single_char": 10}}

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


# ===================== 关键：确保全局实例正确创建 =====================
config_manager = ConfigManager()
get_config = config_manager.get_config
get_tesseract_path = config_manager.get_tesseract_path