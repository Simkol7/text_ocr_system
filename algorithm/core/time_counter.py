"""耗时统计模块：装饰器形式统计函数执行时间，适配毕设性能指标"""
import time
import logging
from functools import wraps
from .logger import logger

def time_counter(func):
    """
    耗时统计装饰器：记录函数执行时间，输出到日志

    Args:
        func: 被装饰的函数

    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = round(end_time - start_time, 4)
            logger.info(f"函数{func.__name__}执行完成，耗时：{elapsed}秒")
            return result
        except Exception as e:
            end_time = time.time()
            elapsed = round(end_time - start_time, 4)
            logger.error(f"函数{func.__name__}执行失败，耗时：{elapsed}秒，异常：{str(e)}")
            raise
    return wrapper