import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('image1.jpg', 0)
img2 = cv2.imread('image2.jpg', 0)

# 特征点检测和匹配
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)

# 选择匹配点对
matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:10]  # 取前10个最佳匹配

# 计算单应性矩阵
src_pts = np.zeros((len(matches), 2), dtype=np.float32)
dst_pts = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    src_pts[i, :] = kp1[match.queryIdx].pt
    dst_pts[i, :] = kp2[match.trainIdx].pt

H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 应用单应性变换
height, width = img1.shape
trans_img = cv2.warpPerspective(img2, H, (width, height))

# 显示结果
cv2.imshow('Original Image', img1)
cv2.imshow('Transformed Image', trans_img)
cv2.waitKey(0)
cv2.destroyAllWindows()