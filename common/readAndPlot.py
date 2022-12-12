# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: readAndPlot.py
@time: 2022-12-10 20:02
"""
import os
import cv2 as cv
import matplotlib.pyplot as plt


def readDir(dir):
    img_path_list = list()
    img_list = list()

    if (os.path.isdir(dir)):
        pathDir = os.listdir(dir);

        for file in pathDir:
            child = os.path.join(dir, file)
            img_path_list.append(child)

    for img_path in img_path_list:
        img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        img_list.append(img)

    return img_list


def show(img_list):
    plt.figure(figsize=(30, 20))

    for i in range(len(img_list)):
        plt.subplot(5, 4, i + 1)
        plt.imshow(img_list[i])

    plt.tight_layout()
    plt.show()
