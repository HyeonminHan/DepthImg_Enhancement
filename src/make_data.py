import cv2
from random import randint
import numpy as np

GT = cv2.imread('../data/r_63_depth_0001.png', 0) # shape (800, 800)
GT_withHole = cv2.imread('../data/r_63_depth_0001.png', 0) # shape (800, 800)
GT_blur = cv2.imread('../data/r_63_depth_0001.png', 0) # shape (800, 800)

## make hole image1 - random
# hole_cnt = GT.shape[0] * GT.shape[1] // 5 # 20% hole
# i_arr = [randint(0, GT.shape[0]-1) for _ in range(hole_cnt)]
# j_arr = [randint(0, GT.shape[1]-1) for _ in range(hole_cnt)]
# for n in range(hole_cnt): 
#     GT_withHole[i_arr[n],j_arr[n]] = 0


## make hole image2 - stride
gt_cnt = GT.shape[0] * GT.shape[1] // 100 # 1% gt
row_cnt = 40
col_cnt = 160

print("row_cnt:", row_cnt)
print("col_cnt:", col_cnt)
bias = 3

i_arr = [GT.shape[0]//row_cnt * i +bias for i in range(row_cnt)]
j_arr = [GT.shape[1]//col_cnt * i +bias for i in range(col_cnt)]

# print("i-arr:", i_arr)
# print("j_arr:", j_arr)
GT_withhole2 = np.zeros(GT.shape, dtype=np.uint8)
for i in i_arr:
    for j in j_arr:
        GT_withhole2[i][j] = GT[i][j]

np.set_printoptions(threshold=np.inf)
print('GT_withhole2', GT_withhole2[500:600])
# cv2.imshow("img2", img1)
# cv2.waitKey(0)


## make blur image
GT_blur = cv2.blur(GT_blur, (10, 10))

# cv2.imshow('GT_withHole', GT_withHole)
cv2.imwrite('../data/GT_withHole2.png', GT_withhole2)
# cv2.imwrite('../data/GT.png', GT)
# cv2.imwrite('../data/blur.png', GT_blur)

exit()