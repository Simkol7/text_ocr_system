"""项目入口文件：一键运行OCR/启动PyQt5界面"""
import sys
import os  # 新增：导入os模块（修复main.py中os未定义的潜在问题）
from algorithm.core import logger, config_manager
from algorithm.input_module.image_loader import load_image

def main():
    """主函数：测试核心模块是否正常运行"""
    logger.info("===== 启动OCR系统 =====")
    try:
        # 测试配置加载
        tesseract_path = config_manager.get_tesseract_path()
        logger.info(f"Tesseract路径配置成功：{tesseract_path}")

        # 测试图像加载（替换为你的测试图像路径）
        test_img_path = "./test/test_cases/test_images/test.jpg"
        if os.path.exists(test_img_path):
            img, scale, success = load_image(test_img_path)
            logger.info(f"测试图像加载成功，缩放系数：{scale}")
        else:
            logger.warning("测试图像不存在，跳过图像加载测试")

        logger.info("===== OCR系统初始化完成 =====")
    except Exception as e:
        logger.error(f"系统启动失败：{str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # 启动PyQt5界面（后续实现）
    # from ui.logic.main_window import MainWindow
    # from PyQt5.QtWidgets import QApplication
    # app = QApplication(sys.argv)
    # window = MainWindow()
    # window.show()
    # sys.exit(app.exec_())

    # 先测试核心模块
    main()