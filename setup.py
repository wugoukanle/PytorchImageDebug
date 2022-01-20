# -*- coding: utf-8 -*-
"""
@Project Name  LocalProject
@File Name:    darknet_networkx
@Software:     PyCharm
@Time:         2020/6/13 10:00
@Author:       taosheng
@contact:      langangpaibian@sina.com
@version:      1.0
@Description:　
"""
from setuptools import setup, find_packages

setup(
    name="pytorch_image_debug",
    version="0.1.0",

    # packages = find_packages(),
    py_modules=["pytorch_image_debug"],
    # scripts = ["pytorch_image_debug.py"],

    install_requires=["numpy", "matplotlib", "Pillow", "torch>=0.3", "torchvision>=0.2"],
    package_data={"": ["*.txt", "*.rst"]},
    python_requires=">=3",  # 指定项目依赖 python版本
    zip_safe=False,

    author="taoshen",
    author_email="langangpaibian.com",
    description="pytorch image debug: image preprocess or tensor show as image",
    long_description="useful torch tensor debug tool",

    keywords=("torch tensor debug", "image preprocess", "image show"),
    license="GPL"
)
