"""全局异常处理模块：统一OCR相关异常定义+友好提示"""
import logging
from .logger import logger

class OCRException(Exception):
    """OCR通用异常基类，所有自定义异常继承此类"""
    def __init__(self, message: str, error_code: int = 1000):
        self.message = message
        self.error_code = error_code
        logger.error(f"OCR异常（{error_code}）：{message}")
        super().__init__(self.message)

class InputError(OCRException):
    """输入相关异常（如路径错误、图像无效）"""
    def __init__(self, message: str, error_code: int = 1001):
        super().__init__(message, error_code)

class PreprocessError(OCRException):
    """预处理相关异常（如CLAHE失败、倾斜校正失败）"""
    def __init__(self, message: str, error_code: int = 1002):
        super().__init__(message, error_code)

class DetectionError(OCRException):
    """检测相关异常（如轮廓提取失败）"""
    def __init__(self, message: str, error_code: int = 1003):
        super().__init__(message, error_code)

class RecognitionError(OCRException):
    """识别相关异常（如Tesseract调用失败）"""
    def __init__(self, message: str, error_code: int = 1004):
        super().__init__(message, error_code)

def handle_ocr_exception(func):
    """OCR异常装饰器：捕获并统一处理函数内的OCR异常"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OCRException as e:
            logger.error(f"函数{func.__name__}执行失败：{e.message}")
            raise  # 重新抛出，让上层处理
        except Exception as e:
            # 非自定义异常转换为通用OCR异常
            logger.error(f"函数{func.__name__}执行异常：{str(e)}", exc_info=True)
            raise OCRException(f"未知异常：{str(e)}", error_code=9999) from e
    return wrapper