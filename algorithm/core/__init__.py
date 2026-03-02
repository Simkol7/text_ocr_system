"""核心工具模块导出，简化外部导入"""
from .logger import logger, setup_logger
from .config_manager import ConfigManager, config_manager, get_config, get_tesseract_path

# 异常类定义（补充OutputError）
class OCRError(Exception):
    """OCR基础异常类"""
    pass

class InputError(OCRError):
    """输入相关异常"""
    pass

class PreprocessError(OCRError):
    """预处理相关异常"""
    pass

class DetectionError(OCRError):
    """检测相关异常"""
    pass

class RecognitionError(OCRError):
    """识别相关异常"""
    pass

class OutputError(OCRError):  # 新增：输出/保存相关异常
    """结果输出/保存相关异常"""
    pass

# 装饰器定义（适配各模块的异常处理）
def handle_ocr_exception(func):
    """OCR通用异常处理装饰器"""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            from .logger import logger
            logger.error(f"函数{func.__name__}执行失败：{str(e)}")
            # 重新抛出对应类型的异常
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"{func.__name__}执行异常：{str(e)}") from e
    return wrapper

# 导出异常类和装饰器（新增OutputError）
__all__ = [
    "logger", "setup_logger",
    "ConfigManager", "config_manager", "get_config", "get_tesseract_path",
    "OCRError", "InputError", "PreprocessError", "DetectionError", "RecognitionError", "OutputError",
    "handle_ocr_exception"
]