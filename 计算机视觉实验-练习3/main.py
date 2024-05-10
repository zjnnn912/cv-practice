import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale
import os

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2, axis=2)
    mse_mean = np.mean(mse)
    if mse_mean == 0:
        return float('inf')

    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_mean))


# 读取原始图像
pic_list = os.listdir('Set5')
for pic in pic_list:
    original_image = cv2.imread('Set5/' + pic)

    # 使用Bicubic插值进行下采样
    downsampled_image = cv2.resize(original_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # 使用scikit-image的rescale函数进行超分辨率处理
    # 这里以放大2倍为例
    super_resolution_image = cv2.resize(downsampled_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # 计算SSIM值以评估超分辨率图像的质量
    original_float = original_image.astype(np.float32) / 255
    super_resolution_image_float = super_resolution_image.astype(np.float32) / 255
    ssim_value = ssim(original_float, super_resolution_image_float, multichannel=True, channel_axis=-1, data_range=1)

    #计算PSNR
    img1 = original_float.astype(np.float64)
    img2 = super_resolution_image.astype(np.float64)
    psnr_value = calculate_psnr(img1, img2)
    # 显示结果
    cv2.imshow(pic + ': ' + 'Original', original_image)
    cv2.imshow(pic + ': ' + 'Downsampled', downsampled_image)
    cv2.imshow(pic + ': ' + 'Super Resolution', super_resolution_image)

    # 等待用户关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(pic + ': ' + f"SSIM value: {ssim_value}")
    print(pic + ': ' + f"PSNR value: {psnr_value}")