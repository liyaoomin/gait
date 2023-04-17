import numpy as np
# import cv2

# print(64//2)
# a = np.arange(27).reshape((3, 3,3))
# print(a)
# print(a[:, :, 0])

# bc=cv2.imread("116-bg-01-018-001.png")
# print(bc.shape)
# c = np.reshape(cv2.imread("116-bg-01-018-001.png"),
#                                  [240,320,-1 ])
# print(c.shape)
# print(c[0, :,:].shape)

# np.reshape(cv2.imread(osp.join(file_path, _img_path)),
#                                  [self.resolution, self.resolution-2*self.cut_padding, -1])


# import os
# import os.path as osp
# import csv
# from copy import deepcopy
#

# conf = {
#     "WORK_PATH": "./work",
#     "CUDA_VISIBLE_DEVICES": "1, 0",
#     "data": {
#         'dataset_path': "../data/GaitDatasetB",  # your_dataset_path
#         'resolution': '128',
#         'dataset': 'CASIA-B',
#         # In CASIA-B, data of subject #5 is incomplete.
#         # Thus, we ignore it in training.
#         'pid_num': 73,
#         'pid_shuffle': False,
#     },
#     "model": {
#         'hidden_dim': 256,
#         'lr': 1e-4,
#         'hard_or_full_trip': 'full',
#         'batch_size': (8, 8),
#         'restore_iter': 0,
#         'total_iter': 150000,
#         'margin': 0.2,
#         'num_workers': 4,
#         'frame_num': 30,
#         'model_name': 'CSTL',
#     },
# }
#
#
# model_config = conf['model']
# data_config = conf['data']
# model_param = deepcopy(model_config)
#
# model_param['save_name'] = '_'.join(map(str,[
#         model_config['model_name'],
#         data_config['dataset'],
#     ]))
#
# # print(model_param['save_name'])
# model_name= conf['model']['model_name']
# save_name = model_param['save_name']
# restore_iter = 1
#
# print(osp.join('checkpoint', model_name,
#             '{}-{:0>5}-encoder.ptm'.format(save_name, restore_iter)))


# import os
# import os.path as osp
# import csv
#
#
# # 读取文件结构并保存为csv
# dataset_path="./data/GaitDatasetB"
# fileHeader = ["_label", "_seq_type", "_view", "seqs"]
#
# csvFile = open("dataset.csv", "w", newline='')
# writer = csv.writer(csvFile)
# writer.writerow(fileHeader)
# for _label in sorted(list(os.listdir(dataset_path))):
#     label_path = osp.join(dataset_path, _label)
#     for _seq_type in sorted(list(os.listdir(label_path))):
#         seq_type_path = osp.join(label_path, _seq_type)
#         for _view in sorted(list(os.listdir(seq_type_path))):
#             _seq_dir = osp.join(seq_type_path, _view)
#             seqs = len(os.listdir(_seq_dir))
#
#             imgs = sorted(list(os.listdir(_seq_dir)))
#
#             frame_list = [ cv2.imread(osp.join(_seq_dir, _img_path)).shape[0:2]
#                           for _img_path in imgs
#                           if osp.isfile(osp.join(_seq_dir, _img_path))]
#
#             writer.writerow([_label, _seq_type, _view, seqs,frame_list])
#
# csvFile.close()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 载入原图，并转为灰度图像
# img_original = cv2.imread('D:/AAAGait/CSTL-main/data/GaitDatasetBB/001/bg-01/000/001-bg-01-000-001.png')
# img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
# # 求二值图像
# retv, thresh = cv2.threshold(img_gray, 125, 255, 1)
# # 寻找轮廓
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # 绘制轮廓
# cv2.drawContours(img_original, contours, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
# # 显示图像
# cv2.imshow('Contours', img_original)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(hierarchy)

#
# import cv2
# import numpy as np
#
# # Load image, grayscale, Gaussian blur, Otsu's threshold
# image = cv2.imread('D:/AAAGait/CSTL-main/data/GaitDatasetBB/001/bg-01/000/001-bg-01-000-074.png')
# original = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# # Find contours
# ROI_number = 0
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     # Obtain bounding rectangle to get measurements
#     x,y,w,h = cv2.boundingRect(c)
#
#     # Find centroid
#     M = cv2.moments(c)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#
#     # Crop and save ROI
#     ROI = original[y:y+h, x:x+w]
#     cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
#     ROI_number += 1
#
#     # Draw the contour and center of the shape on the image
#     cv2.rectangle(image,(x,y),(x+w,y+h),(36,255,12), 4)
#     cv2.circle(image, (cX, cY), 10, (320, 159, 22), -1)
#
# cv2.imwrite('image1.png', image)
# cv2.imwrite('thresh1.png', thresh)
# cv2.waitKey()
#
#
# import cv2
#
# # Load image, grayscale, Gaussian blur, Otsu's threshold
# image = cv2.imread('D:/AAAGait/CSTL-main/data/GaitDatasetBB/001/cl-01/036/001-cl-01-036-086.png')
#
# original = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# old_size = image.shape[0:2]  # 原始图像大小
# target_size = [128, 88]  # 目标图像大小
#
# # Find contours
# ROI_number = 0
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     # Obtain bounding rectangle to get measurements
#     x,y,w,h = cv2.boundingRect(c)
#
#     # Crop and save ROI
#     ROI = original[y:y+h, x:x+w]
#
#     old_size = ROI.shape[0:2]  # 原始图像大小
#
#     ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
#     new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
#
#     img = cv2.resize(ROI,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
#     pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
#     pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
#     top,bottom = pad_h//2, pad_h-(pad_h//2)
#     left,right = pad_w//2, pad_w -(pad_w//2)
#     img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
#     cv2.imwrite('img_new.png', img_new)
#     cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
#     ROI_number += 1

import os
import cv2


def crop_ROI(img_dir):
    areas = []
    image = cv2.imread(img_dir)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in range(len(contours)):
    #     areas.append(cv2.contourArea(contours[c]))
    #
    # if len(areas) == 0:
    #     ROI = thresh
    # else:
    #     max_id = areas.index(max(areas))
    #     x, y, w, h = cv2.boundingRect(contours[max_id])
    #     ROI = original[y:y + h, x:x + w]

    if len(contours) == 0:
        ROI = thresh
    else:
        for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))
        max_id = areas.index(max(areas))
        x, y, w, h = cv2.boundingRect(contours[max_id])
        ROI = original[y:y + h, x:x + w]

    return ROI


def resize_img_keep_ratio(image, target_size):
    old_size = image.shape[0:2]  # 原始图像大小
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(image, (new_size[1], new_size[0]))  # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


data_dir = "./data/GaitDatasetB"
target_size = [128, 88]  # 目标图像大小
for root, dirs, files in os.walk(data_dir):
    for name in files:
        if name.endswith(".png"):
            img_dir = os.path.join(root, name)
            img = cv2.imread(img_dir)
            ROI = crop_ROI(img_dir)
            resize_img = resize_img_keep_ratio(ROI, target_size)
            cv2.imwrite(img_dir, resize_img)

