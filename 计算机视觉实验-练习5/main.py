import cv2
import numpy as np

# 载入左右图像
left_img = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)

# 相机标定参数
camera_matrix = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
dist_coeffs = np.array([[0, 0, 0, 0, 0]])

# 图像校正
left_img_undistorted = cv2.undistort(left_img, camera_matrix, dist_coeffs)
right_img_undistorted = cv2.undistort(right_img, camera_matrix, dist_coeffs)

# 立体校正参数（示例值，需要替换为实际计算结果）
R = np.eye(3)
T = np.array([0, 0, 0])
R1, R2, P1, P2, Q = cv2.stereoRectify(camera_matrix, dist_coeffs, None, None, left_img.shape, R, T)

# 计算视差图
block_size = 21
min_disp = 0
num_disp = 64
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=block_size,
                               P1=8 * 3 * block_size ** 2,
                               P2=32 * 3 * block_size ** 2,
                               disp12MaxDiff=1,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32)

disparity = stereo.compute(left_img_undistorted, right_img_undistorted)[1]

# 视差图优化
disparity = cv2.medianBlur(disparity, 5)


# 可视化视差图
cv2.imshow('Disparity Map', disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()