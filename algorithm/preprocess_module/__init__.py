"""预处理模块：灰度化、CLAHE增强、降噪、二值化、形态学优化、倾斜校正"""
# 快捷导入核心函数/类，简化上层调用
from .gray_process import to_gray
from .clahe_enhance import clahe_enhance
from .noise_removal import remove_noise
from .binarization import binarize
from .morphology_optim import morphology_optimize
from .orientation_fix import fix_orientation
from .preprocess_core import run_preprocess

# 预处理流程顺序（常量，方便维护）
PREPROCESS_STEPS = ["gray", "clahe", "noise_removal", "binarization", "morphology", "orientation_fix"]