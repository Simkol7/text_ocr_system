import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name: str = "text_ocr_system") -> logging.Logger:
    """
    配置滚动日志器：单个文件最大10MB，最多保留5个备份，避免日志体积爆炸
    适配批量测试场景，符合毕设工程化规范
    """
    # 日志保存目录（自动创建）
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 滚动日志文件名（固定名称，自动滚动）
    log_file = os.path.join(log_dir, "ocr_system.log")

    # 日志格式：时间-日志器-级别-模块-行号-消息
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 初始化日志器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # 避免重复添加handler

    # 1. 控制台handler（INFO及以上）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 2. 替换为滚动文件handler（DEBUG及以上，批量测试友好）
    # 配置：maxBytes=10*1024*1024（10MB），backupCount=5（保留5个备份）
    file_handler = RotatingFileHandler(
        log_file,
        mode="a",          # 追加模式
        maxBytes=10*1024*1024,  # 单个文件最大10MB
        backupCount=5,     # 最多保留5个备份文件
        encoding="utf-8"   # 适配中文日志
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.info(f"滚动日志器初始化完成，日志文件：{log_file}，单个文件最大10MB，保留5个备份")
    return logger

# 全局日志器实例
logger = setup_logger()