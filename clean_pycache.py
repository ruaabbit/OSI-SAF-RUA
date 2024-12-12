"""清除 __pycache__ 缓存"""

import os
import shutil


def purge_cache(path):
    # 遍历目录下所有文件
    for file_name in os.listdir(path):
        abs_path = os.path.join(path, file_name)
        if file_name == "__pycache__":
            print(abs_path)
            # 删除 `__pycache__` 目录及其中的所有文件
            shutil.rmtree(abs_path)
        elif os.path.isdir(abs_path):
            # 递归调用
            purge_cache(abs_path)


if __name__ == "__main__":
    root_dir = r"D:\Seaice\OSI-SAF-RUA"

    purge_cache(root_dir)
