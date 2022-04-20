# import OpenEXR ## conda-forge openexr-python install
# golden = OpenEXR.InputFile('/media/hyunmin/FED62A69D62A21FF/Depth Enhancement/lidar_depth/0.exr')
# print(golden.header())

import imageio
import numpy as np
# imageio.plugins.freeimage.download()

#shape 1280 * 720
cnt_l = 0
cnt_z = 0
for i in range(179):
    fname_lidar = '/media/hyunmin/FED62A69D62A21FF/Depth Enhancement/lidar_depth/' + str(i) + '.exr'
    fname_zed = '/media/hyunmin/FED62A69D62A21FF/Depth Enhancement/zed_depth/' + str(i) + '.exr'
    exr_lidar = imageio.imread(fname_lidar, format='EXR-FI')
    exr_zed = imageio.imread(fname_zed, format='EXR-FI')
    exr_lidar = np.array(exr_lidar)
    exr_zed = np.array(exr_zed)
    # np.set_printoptions(threshold=np.inf)
    # print(exr_zed)
    print('i:', i)
    print('min lidar: ', exr_lidar[exr_lidar>0].min()) # 0.0
    print('max lidar: ', exr_lidar[exr_lidar>0].max()) # 0 - 1.8274628 /  50 - 3.49 / 60 - 6.65 / 70 - 11.04 /  80 - 11.11 / 90 - 11.09 / 100 - 11.04..
    print('min zed: ', exr_zed[exr_zed>0].min()) # 0.0
    print('max zed: ', exr_zed[exr_zed>0].max()) 

    if exr_zed[exr_zed>0].max() > 5:
        cnt_z +=1
    if exr_lidar[exr_lidar>0].max() > 5:
        cnt_l +=1
    print('not hole cnt :', len(exr_lidar[np.where(exr_lidar>0)]))
    print('0<pixel< 3cnt :', len(exr_lidar[np.where(np.bitwise_and(exr_lidar>0, exr_lidar<3))]))

    
    print('gt percentage lidar: ', len(exr_lidar[exr_lidar>0]) / (exr_lidar.shape[0] * exr_lidar.shape[1]) * 100, ' gt percentage zed: ', len(exr_zed[exr_zed>0]) / (exr_zed.shape[0] * exr_zed.shape[1]) * 100) # 8223 1% 이내...
    print('--------------')

print('cnt_l:', cnt_l)
print('cnt_z:', cnt_z)
