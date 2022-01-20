# -*- coding: utf-8 -*-
"""
@Project Name  learn_pytorch
@File Name:    util
@Software:     PyCharm
@Time:         2019/5/30 14:43
@Author:       taosheng
@contact:      langangpaibian@sina.com
@version:      1.0
@Description:　
"""
import os
import sys
from pathlib import Path
import platform

if platform.system() == "Linux":
    sys.path.append((Path(os.path.abspath(__file__)).parents[1]).as_posix())
    print("sys.path.append: ", (Path(os.path.abspath(__file__)).parents[1]).as_posix())

import math
import numpy as np
import pandas as pd

import matplotlib
if platform.system() == "Linux":
    matplotlib.use('agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

import torch
from torchvision.utils import *

from colorama import init, Fore, Back, Style

if "PYCHARM_HOSTED" in os.environ:
    convert = False
    strip = False
else:
    convert = None
    strip = None

init(convert=convert, strip=strip)

np.set_printoptions(threshold=20000, linewidth=20000)  # default 1000 default 75
np.set_printoptions(suppress=True)

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

cur_pdir = Path(os.path.dirname(__file__))
print("cur_pdir: ", cur_pdir)


class ImageRange(object):
    def __init__(self):
        pass

    @staticmethod
    def image_range_255(img, ignore_range=True):
        """
        将浮点型图像转化为整数型型图像
        :param img: 数据类型为ndarray
        :return:
        """
        assert str(img.dtype) == "float32"
        assert img.min() >= 0 and img.max() <= 1.
        print("type(img): ", type(img))
        print("img.min(), img.max(): ", img.min(), img.max())

        if ignore_range:
            rescale_intensity(img, in_range="image", out_range=(0, 255))
            img = img.astype(np.uint8)
        else:
            rescale_intensity(img, in_range=(0, 1.0), out_range=(0, 255))
            img = img.astype(np.uint8)

        print("type(img): ", type(img))
        print("img.min(), img.max(): ", img.min(), img.max())

        return img

    @staticmethod
    def image_range_1(img, ignore_range=True):
        """
        将整数型图像转化为浮点型图像
        :param img: 数据类型为ndarray
        :return:
        """
        assert isinstance(img, np.uint8)
        assert img.min() >= 0 and img.max() <= 255

        if ignore_range:
            img = img.astype(np.float32)
            img = rescale_intensity(img, in_range="image", out_range=(0, 1.0))
        else:
            img = img.astype(np.float32)
            img = rescale_intensity(img, in_range=(0, 255), out_range=(0, 1.0))

        return img


class ImagePreprocess(object):

    @staticmethod
    def get_mean_std(root_path):
        """求图像通道的均值和方差"""
        path_list = os.listdir(root_path)
        img_mean_list = []
        img_std_list = []
        for c in range(3):
            img_list = []
            for img in path_list:
                img_path = (Path(root_path) / img).as_posix()
                Img = Image.open(img_path).convert("RGB")
                img = np.array(Img)
                img = img.astype(np.float32)
                img = rescale_intensity(img, in_range=(0, 255), out_range=(0, 1.0))
                # print("img.shape: ", img.shape)

                img_list.append(img[:, :, c].ravel())
            img_con = np.concatenate(img_list, axis=0)
            img_mean_list.append(np.mean(img_con, axis=0))
            img_std_list.append(np.std(img_con, axis=0))

        return img_mean_list, img_std_list

    @staticmethod
    def locate_crop(img, mask, square_size=560):
        """
        对目标进行crop
        :param img: ndarray
        :param mask: ndarray
        :return:
        """
        assert img.shape == mask.shape

        image_width = img.shape[1]
        image_height = img.shape[0]

        Img = Image.fromarray(img)
        Mask = Image.fromarray(mask)

        left, upper, right, lower = Mask.getbbox()

        y1 = upper
        x1 = left
        y2 = lower
        x2 = right

        width = x2 - x1
        height = y2 - y1

        ##################################################
        if width < square_size:
            if (square_size - width) % 2 == 0:
                x1 -= (square_size - width) // 2
                x2 += (square_size - width) // 2
            else:
                x1 -= (square_size - width) // 2
                x2 += (square_size - width) // 2 + 1
        else:
            raise Exception("square size error")

        if height < square_size:
            if (square_size - height) % 2 == 0:
                y1 -= (square_size - height) // 2
                y2 += (square_size - height) // 2
            else:
                y1 -= (square_size - height) // 2
                y2 += (square_size - height) // 2 + 1
        else:
            raise Exception("square size error")

        ##################################################
        if y1 < 0:
            y2 += abs(y1)
            y1 = 0

        if x1 < 0:
            x2 += abs(x1)
            x1 = 0

        if y2 > (image_height - 1):
            y1 -= (y2 - image_height + 1)
            y2 = image_height - 1

        if x2 > (image_width - 1):
            x1 -= (x2 - image_width + 1)
            x2 = image_width - 1

        assert (x2 - x1) == square_size
        assert (y2 - y1) == square_size

        Img_crop = Img.crop((x1, y1, x2, y2))
        Mask_crop = Mask.crop((x1, y1, x2, y2))

        return np.array(Img_crop), np.array(Mask_crop)


class ImageBox(object):
    def __init__(self):
        pass

    @staticmethod
    def get_box_list(img_shape, win_shape):
        """
        得到所有box坐标
        :param img_shape:
        :param win_shape:
        :return:
        """
        if len(img_shape) == 1:
            img_height = img_shape[0]
            img_width = img_shape[0]
        elif len(img_shape) == 2:
            img_height, img_width = img_shape
        else:
            raise Exception("img_shape error")

        if len(win_shape) == 1:
            win_height = win_shape[0]
            win_width = win_shape[0]
        elif len(win_shape) == 2:
            win_height, win_width = win_shape
        else:
            raise Exception("win_shape error")

        if win_height > img_height or win_width > img_width:
            raise Exception("size error")

        # 列方向最大多少个batch
        if img_height % win_height == 0:
            row_num = img_height // win_height
        else:
            row_num = img_height // win_height + 1

        # 行方向最大多少个batch
        if img_width % win_width == 0:
            col_num = img_width // win_width
        else:
            col_num = img_width // win_width + 1

        # 生成列方向的采样点
        samples, step = np.linspace(0, (img_height - win_height), row_num, retstep=True)
        if math.isnan(step):
            row_sample = [0]
        elif math.modf(step) == 0:
            row_sample = samples
        else:
            row_sample = list(range(0, (img_height - win_height), int(step) + 1))
            row_sample.append(img_height - win_height)
        assert len(row_sample) == row_num
        # print("row_sample:", row_sample)

        # 生成行方向的采样点
        samples, step = np.linspace(0, (img_width - win_width), col_num, retstep=True)
        if math.isnan(step):
            col_sample = [0]
        elif math.modf(step) == 0:
            col_sample = samples
        else:
            col_sample = list(range(0, (img_width - win_width), int(step) + 1))
            col_sample.append(img_width - win_width)
        assert len(col_sample) == col_num
        # print("col_sample:", col_sample)

        # 根据得到的左上角坐标生成box的坐标
        from itertools import product
        row_col_lu = product(row_sample, col_sample)
        box_list = []
        for y1, x1 in row_col_lu:
            box_list.append((x1, y1, x1 + win_width, y1 + win_height))

        return box_list

    @staticmethod
    def extract_window(img, win_shape):
        """
        提取图像子窗口
        :param img:
        :param win_shape:
        :return:
        """
        assert isinstance(win_shape, tuple)

        box_list = ImageBox.get_box_list(img.shape[0:2], win_shape)

        img_pil = Image.fromarray(img)
        img_list = []
        for box in box_list:
            block = img_pil.crop(box)
            img_list.append(np.array(block))

        return img_list

    @staticmethod
    def concate_window(img_list, img_shape):
        """
        使用最大值融合不同patch图像
        :param img_shape: 不包括通道数
        :param img_list:
        :param box_list:
        :return:
        """
        assert len(img_list) > 0

        box_list = ImageBox.get_box_list(img_shape, img_list[0].shape)

        img_num = len(img_list)
        img_stack = np.zeros((*img_shape, img_num))

        for i, img in enumerate(img_list):
            img_slice = img_stack[:, :, i]
            x1, y1, x2, y2 = box_list[i]
            img_slice[y1:y2, x1:x2] = img

        img_fuse = img_stack.max(axis=2)

        return img_fuse


class ImageDisplay(object):
    def __init__(self):
        pass

    @staticmethod
    def show_image(image, img_ppath=Path('./temp.png'), max_val=255):
        """
        显示tensor image用于调试
        :param image:
        :return:
        """
        print("type(image): ", type(image))
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.cpu().numpy().transpose((1, 2, 0))
            elif len(image.shape) == 2:
                image = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            if str(image.dtype) == "bool":
                image = image.astype(np.uint8)
        else:
            raise Exception("image type error")
        print("image.shape: ", image.shape)
        print("image.dtype: ", image.dtype)

        image = rescale_intensity(image, in_range='image', out_range=(0, max_val))
        image = image.astype(np.uint8)
        print("image.dtype: ", image.dtype)
        print("image.min(), image.max(): ", image.min(), image.max())

        if len(image.shape) == 2:
            cmap = plt.cm.gray
        else:
            cmap = None

        plt.figure()
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
        if platform.system() == "Windows":
            plt.show()
        else:
            plt.savefig(img_ppath.as_posix())

    @staticmethod
    def show_image_batch(images_batch, img_ppath=Path('./temp.png')):
        """
        显示一个batch中所有的图像
        :param images_batch: B x C x H x W
        :return:
        """
        if len(images_batch.shape) == 3:
            images_batch = images_batch[:, np.newaxis, :, :]

        grid = make_grid(images_batch, nrow=4, padding=0, normalize=True, range=None, scale_each=True, pad_value=0)

        cmap = None
        if len(images_batch.shape) == 4:
            grid = grid.cpu().numpy().transpose((1, 2, 0))
        elif len(images_batch.shape) == 3:
            grid = grid.cpu().numpy()
            cmap = plt.cm.gray
        else:
            raise Exception("batch shape error")

        plt.figure()
        plt.imshow(grid, cmap=cmap)
        plt.axis('off')
        if platform.system() == "Windows":
            plt.show()
        else:
            plt.savefig(img_ppath.as_posix())

    @staticmethod
    def plot_img_and_mask(image, mask, img_ppath=Path('./temp.png')):
        """
        显示图像及对应的mask
        :param image:
        :param mask:
        :return:
        """
        print("type(image): ", type(image))
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            if len(image.shape) == 3:
                image = image.cpu().numpy().transpose((1, 2, 0))
                mask = mask.cpu().numpy().transpose((1, 2, 0))
            elif len(image.shape) == 2:
                image = image.cpu().numpy()
                mask = mask.cpu().numpy()
        elif isinstance(image, np.ndarray) and isinstance(mask, np.ndarray):
            pass
        elif isinstance(image, Image.Image) and isinstance(mask, Image.Image):
            image = np.array(image)
            mask = np.array(mask)
        else:
            raise Exception("image type error")
        print("image.shape: ", image.shape)
        print("image.dtype: ", image.dtype)
        print("mask.shape: ", mask.shape)
        print("mask.dtype: ", mask.dtype)

        image = rescale_intensity(image, in_range='image', out_range=(0, 255))
        image = image.astype(np.uint8)

        mask = rescale_intensity(mask, in_range='image', out_range=(0, 255))
        mask = mask.astype(np.uint8)

        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        a.set_title('Input image')
        plt.imshow(image)

        b = fig.add_subplot(1, 2, 2)
        b.set_title('Output mask')
        plt.imshow(mask, cmap='gray')
        if platform.system() == "Windows":
            plt.show()
        else:
            plt.savefig(img_ppath.as_posix())

    @staticmethod
    def plot_img_list(img_list, same_figure=True, img_ppath=Path('./temp.png')):

        if same_figure:
            ## 显示在同一幅图中 ##
            fig = plt.figure()

            img_num = len(img_list)
            for i, img in enumerate(img_list):
                a = fig.add_subplot(1, img_num, i + 1)
                a.set_title('Input image {}'.format(i))
                a.axes.imshow(img)
                plt.axis("off")

            if platform.system() == "Windows":

                plt.show()
            else:
                plt.savefig(img_ppath.as_posix())

        else:
            ## 显示在不同图中 ##
            for i, img in enumerate(img_list):
                plt.figure()
                plt.title('Input image {}'.format(i))
                plt.imshow(img)
                plt.axis("off")
                if platform.system() == "Windows":
                    plt.show()
                else:
                    plt.savefig((img_ppath.parent / (img_ppath.stem + f"_{i}" + img_ppath.suffix)).as_posix())

    @staticmethod
    def plot_pixel_hist(image, img_ppath=Path('./temp.png')):
        """
        绘制图像像素值直方图
        :param image:
        :return:
        """
        print("type(image): ", type(image))
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.cpu().numpy().transpose((1, 2, 0))
            elif len(image.shape) == 2:
                image = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            pass
        else:
            raise Exception("image type error")

        print("image.min(), image.max(): ", image.min(), image.max())

        plt.figure()
        plt.hist(image.ravel(), bins=256, fc="k", ec="k")
        plt.axis("off")
        if platform.system() == "Windows":
            plt.show()
        else:
            plt.savefig(img_ppath.as_posix())


if __name__ == "__main__":
    print("os.getcwd(): ", os.getcwd())
