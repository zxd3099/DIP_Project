# encoding: utf-8
"""
@version: 1.0
@author: zxd3099
@file: IndexCalc
@time: 2022-12-10 20:12
"""
import math
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR


"""
Full-reference metrics
"""


def calc_psnr(image1, image2):
    """
    PSNR（Peak Signal to Noise Ratio）is the ratio of the signal maximum power to the signal noise power.
    :param image1: original image
    :param image2: enhanced image
    :return:
    """
    psnr = PSNR(image1, image2)
    return psnr


def calc_ssim(image1, image2):
    """
    SSIM (Structural Similarity) is a measure of the similarity between two images.
    The range of SSIM is from -1 to 1. The value of SSIM is equal to 1 when the two images are identical.
    :param image1: original image
    :param image2: enhanced image
    :return:
    """
    ssim =  SSIM(image1, image2, multichannel=True)
    return ssim


"""
No-reference metrics
"""


def calc_eme(image, L = 5):
    """
    EME(Enhancement Measure Evaluation) is the performance of the change degree of the local gray level of the image,
    the larger the EME, the stronger the details of the image performance
    :param image: input image
    :param L: block size
    :return:
    """
    m, n, c = np.shape(image)
    number_m = math.floor(m / L)
    number_n = math.floor(n / L)

    m1 = 0
    E = 0
    for i in range(number_m):
        n1 = 0
        for j in range(number_n):
            A1 = image[m1: m1 + L, n1: n1 + L]
            image_min = np.amin(np.amin(A1))
            image_max = np.amax(np.amax(A1))

            if image_min > 0:
                image_ratio = image_max / image_min
            else:
                image_ratio = image_max
            E = E + np.log(image_ratio + 1e-5)

            n1 = n1 + L
        m1 = m1 + L
    E_sum = 2 * E / (number_m * number_n)
    return E_sum


def calc_loe(image, image1):
    """
    LOE(Lightness Order Error) reflects the natural preservation ability of the reflected image, the smaller the value,
    the better the brightness order of the image, which looks more natural.
    :param image: input rgb image
    :param image1: enhanced rgb image
    :return:
    """
    n, N, M = np.shape(image)

    L = np.max(image, axis=0)
    Le = np.max(image1, axis=0)
    RD = np.zeros([N, M])

    for y in range(1, M):
        for x in range(1, N):
            E = np.logical_xor((L[x, y] >= L[:, :]), (Le[x, y] >= Le[:, :]))
            RD[x, y] = np.sum(E)

    LOE = np.sum(RD) / (M * N)
    return LOE
