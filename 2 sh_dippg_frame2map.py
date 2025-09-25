import csv
import cv2
import os
import numpy as np
import pandas as pd
import xlrd
import face_recognition
from PIL import Image

"""
01_ubfc, 02_cohface, 03_vipl, 04_pure, 05_hci,
06_nirp_indoor, 07_nirp_car940, 08_nirp_car975,
09_buaa, 10_tokyo, 11_vv2, 13_mmpd, 14_yawdd
"""

data_name = '07_nirp_car940'
frame_name = 'Frame_NIR'

train_dataset = []

if data_name == '07_nirp_car940':
    data_total_xls_path = '../rppg_data/07_NIRP_Car940_Dataset/Car_940_data_total.xls'

    # data_total_xls_path = '../rppg_data/07_NIRP_Car940_Dataset/Car_975_data_garage_still.xls'
    # data_total_xls_path = '../rppg_data/07_NIRP_Car940_Dataset/Car_975_data_garage_small_motion.xls'
    # data_total_xls_path = '../rppg_data/07_NIRP_Car940_Dataset/Car_975_data_garage_large_motion.xls'
    # data_total_xls_path = '../rppg_data/07_NIRP_Car940_Dataset/Car_975_data_driving_still.xls'
    # data_total_xls_path = '../rppg_data/07_NIRP_Car940_Dataset/Car_975_data_driving_small_motion.xls'
    # data_total_xls_path = '../rppg_data/07_NIRP_Car940_Dataset/Car_975_data_driving_large_motion.xls'

elif data_name == '08_nirp_car975':
    data_total_xls_path = '../rppg_data/08_NIRP_Car975_Dataset/Car_975_data_total.xls'

data_total_xls = xlrd.open_workbook_xls(data_total_xls_path)
sheet = 'sheet'
sheet_data = data_total_xls.sheet_by_name(sheet)
rows_data = sheet_data.nrows

for i in range(rows_data):
    x = sheet_data.cell(i, 0).value
    x = '../rppg_data/' + x[13:]
    train_dataset.append(x)


def prevent_empty(x):
    assert len(x.shape) == 3

    if x.shape[0] == 1:

        if x.shape[1] == 1:
            y = np.zeros((2, 1, 3))
            y[0, 0, :] = x[0, 0, :]
            y[1, 0, :] = x[0, 0, :]

        else:
            y = x
    else:
        y = x

    return y


def get_img_face(img_raw, face_location):

    if face_location[0][0] < 1:
        y_0 = 1

    else:
        y_0 = face_location[0][0]

    if face_location[0][1] > img_raw.shape[1] - 1:
        x_1 = img_raw.shape[1] - 1

    else:
        x_1 = face_location[0][1]

    if face_location[0][2] > img_raw.shape[0] - 1:
        y_1 = img_raw.shape[0] - 1

    else:
        y_1 = face_location[0][2]

    if face_location[0][3] < 1:
        x_0 = 1

    else:
        x_0 = face_location[0][3]

    img_face = img_raw[y_0: y_1, x_0: x_1]

    region_0 = img_raw[0: y_0, 0: x_0]
    region_1 = img_raw[0: y_0, x_0: x_1]
    region_2 = img_raw[0: y_0, x_1: img_raw.shape[1]]
    region_3 = img_raw[y_0: y_1, x_1: img_raw.shape[1]]
    region_4 = img_raw[y_1: img_raw.shape[0], x_1: img_raw.shape[1]]
    region_5 = img_raw[y_1: img_raw.shape[0], x_0: x_1]
    region_6 = img_raw[y_1: img_raw.shape[0], 0: x_0]
    region_7 = img_raw[y_0: y_1, 0: x_0]

    region_0 = prevent_empty(region_0)
    region_1 = prevent_empty(region_1)
    region_2 = prevent_empty(region_2)
    region_3 = prevent_empty(region_3)
    region_4 = prevent_empty(region_4)
    region_5 = prevent_empty(region_5)
    region_6 = prevent_empty(region_6)
    region_7 = prevent_empty(region_7)

    region_0 = cv2.resize(region_0, (2, 4))
    region_1 = cv2.resize(region_1, (2, 4))
    region_2 = cv2.resize(region_2, (2, 4))
    region_3 = cv2.resize(region_3, (2, 4))
    region_4 = cv2.resize(region_4, (2, 4))
    region_5 = cv2.resize(region_5, (2, 4))
    region_6 = cv2.resize(region_6, (2, 4))
    region_7 = cv2.resize(region_7, (2, 4))

    img_back = np.zeros((1, 64, 3))

    k = 0 * 8
    for i in range(region_0.shape[0]):
        for j in range(region_0.shape[1]):
            img_back[0, k, 0] = region_0[i, j, 0]
            img_back[0, k, 1] = region_0[i, j, 1]
            img_back[0, k, 2] = region_0[i, j, 2]
            k += 1

    k = 1 * 8
    for i in range(region_0.shape[0]):
        for j in range(region_0.shape[1]):
            img_back[0, k, 0] = region_1[i, j, 0]
            img_back[0, k, 1] = region_1[i, j, 1]
            img_back[0, k, 2] = region_1[i, j, 2]
            k += 1

    k = 2 * 8
    for i in range(region_0.shape[0]):
        for j in range(region_0.shape[1]):
            img_back[0, k, 0] = region_2[i, j, 0]
            img_back[0, k, 1] = region_2[i, j, 1]
            img_back[0, k, 2] = region_2[i, j, 2]
            k += 1

    k = 3 * 8
    for i in range(region_0.shape[0]):
        for j in range(region_0.shape[1]):
            img_back[0, k, 0] = region_3[i, j, 0]
            img_back[0, k, 1] = region_3[i, j, 1]
            img_back[0, k, 2] = region_3[i, j, 2]
            k += 1

    k = 4 * 8
    for i in range(region_0.shape[0]):
        for j in range(region_0.shape[1]):
            img_back[0, k, 0] = region_4[i, j, 0]
            img_back[0, k, 1] = region_4[i, j, 1]
            img_back[0, k, 2] = region_4[i, j, 2]
            k += 1

    k = 5 * 8
    for i in range(region_0.shape[0]):
        for j in range(region_0.shape[1]):
            img_back[0, k, 0] = region_5[i, j, 0]
            img_back[0, k, 1] = region_5[i, j, 1]
            img_back[0, k, 2] = region_5[i, j, 2]
            k += 1

    k = 6 * 8
    for i in range(region_0.shape[0]):
        for j in range(region_0.shape[1]):
            img_back[0, k, 0] = region_6[i, j, 0]
            img_back[0, k, 1] = region_6[i, j, 1]
            img_back[0, k, 2] = region_6[i, j, 2]
            k += 1

    k = 7 * 8
    for i in range(region_0.shape[0]):
        for j in range(region_0.shape[1]):
            img_back[0, k, 0] = region_7[i, j, 0]
            img_back[0, k, 1] = region_7[i, j, 1]
            img_back[0, k, 2] = region_7[i, j, 2]
            k += 1

    face_resize = cv2.resize(img_face, (8, 8))
    face_map = np.zeros((1, 64, 3))
    m = 0
    for i in range(face_resize.shape[0]):
        for j in range(face_resize.shape[0]):
            face_map[0, m, 0] = face_resize[i, j, 0]
            face_map[0, m, 1] = face_resize[i, j, 1]
            face_map[0, m, 2] = face_resize[i, j, 2]
            m += 1

    glob_resize = cv2.resize(img_raw, (8, 8))
    glob_map = np.zeros((1, 64, 3))
    n = 0
    for i in range(glob_resize.shape[0]):
        for j in range(glob_resize.shape[0]):
            glob_map[0, n, 0] = glob_resize[i, j, 0]
            glob_map[0, n, 1] = glob_resize[i, j, 1]
            glob_map[0, n, 2] = glob_resize[i, j, 2]
            n += 1

    return face_map, glob_map, img_back


for j in range(len(train_dataset)):
    frame_dir = train_dataset[j] + frame_name + '/'
    frame_list = os.listdir(frame_dir)
    frame_list.sort()

    for frame_one in frame_list:
        frame_path = frame_dir + frame_one

        if frame_path.endswith('.db'):
            os.remove(frame_path)

    frame_list = os.listdir(frame_dir)
    frame_list.sort()

    frame_map = np.zeros((len(frame_list), 64, 3))
    face_map = np.zeros((len(frame_list), 64, 3))
    back_map = np.zeros((len(frame_list), 64, 3))

    for k in range(len(frame_list)):
        img_path = frame_dir + frame_list[k]
        print(img_path)
        img_raw = cv2.imread(img_path)
        img_pil = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        if k == 0:
            face_location = face_recognition.face_locations(img_pil, model='cnn')

            if len(face_location) != 0:
                have_face_path = img_path

            else:
                img_temporary = cv2.imread(have_face_path)
                img_temporary = cv2.cvtColor(img_temporary, cv2.COLOR_BGR2RGB)
                face_location = face_recognition.face_locations(img_temporary, model='cnn')

            have_face_location = face_location

            x_spn = face_location[0][1] - face_location[0][3]
            y_spn = face_location[0][2] - face_location[0][0]

            min_last_spn = np.min([x_spn, y_spn])

            single_face, single_glob, single_back = get_img_face(img_raw, face_location)

            face_map[k, :, :] = single_face[0, :, :]
            frame_map[k, :, :] = single_glob[0, :, :]
            back_map[k, :, :] = single_back[0, :, :]

        else:
            face_location = face_recognition.face_locations(img_pil, model='hog')

            if len(face_location) == 0:
                face_location = have_face_location

            else:
                x_this_spn = face_location[0][1] - face_location[0][3]
                y_this_spn = face_location[0][2] - face_location[0][0]
                min_this_spn = np.min([x_this_spn, y_this_spn])

                if min_this_spn > (0.8 * min_last_spn):
                    have_face_location = face_location
                    min_last_spn = min_this_spn

                else:
                    face_location = have_face_location

            single_face, single_glob, single_back = get_img_face(img_raw, face_location)

            face_map[k, :, :] = single_face[0, :, :]
            frame_map[k, :, :] = single_glob[0, :, :]
            back_map[k, :, :] = single_back[0, :, :]

    cv2.imwrite(train_dataset[j] + 'node_frame_rgb_for_traffic_task.png', frame_map)
    cv2.imwrite(train_dataset[j] + 'node_face_rgb_for_traffic_task.png', face_map)
