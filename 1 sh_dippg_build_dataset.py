import csv
import os
import cv2
import matplotlib.pyplot as plt
import scipy.signal
import xlrd
import xlwt
import numpy as np
import math


""" 01_ubfc, 02_cohface, 03_vipl, 04_pure, 05_hci,
    06_nirp_indoor, 07_nirp_car940, 08_nirp_car975, 09_buaa, 10_Tokyo, 11_VIPL_V2 """

raw_dataset_name = '07_nirp_car940'

window_size = 320


def max_min_norm(x):
    if np.max(x) == np.min(x):
        y = (x - np.min(x)) * 255
    else:
        y = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
    return y


def node_map_norm(x):
    if len(x.shape) == 3:
        node_norm = np.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=np.uint8)
        for i in range(x.shape[0]):
            node_norm[i, :, 0] = max_min_norm(x[i, :, 0])
            node_norm[i, :, 1] = max_min_norm(x[i, :, 1])
            node_norm[i, :, 2] = max_min_norm(x[i, :, 2])
    elif len(x.shape) == 2:
        node_norm = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
        for i in range(x.shape[0]):
            node_norm[i, :] = max_min_norm(x[i, :])
    return node_norm


def generative_smooth_bvp(x, line_top):
    y = np.zeros((len(x)))
    for i in range(len(line_top) - 1):
        l = line_top[i + 1] - line_top[i]
        if l > 2:
            for k in range(l):
                w = l
                w = ((k / w) * 2) * math.pi
                w = math.cos(w)
                w = (w + 1) * 0.5
                y[line_top[i] + k] = w + 0.00001

    if line_top[0] != 0:
        if line_top[0] != 1:
            x_1 = x[0:line_top[0]]
            y[0:line_top[0]] = (x_1 - np.min(x_1)) / (np.max(x_1) - np.min(x_1))

    if line_top[len(line_top) - 1] != len(x):
        if line_top[len(line_top) - 1] != len(x) - 1:
            x_2 = x[line_top[len(line_top) - 1]: len(x)]
            y[line_top[len(line_top) - 1]: len(y)] = (x_2 - np.min(x_2)) / (np.max(x_2) - np.min(x_2))
    return y


def rgb2yuv(x):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2YUV)
    x = node_map_norm(x)
    return x


def gen_wave_map(x):
    x = max_min_norm(x)
    label_map = np.zeros((128, len(x)), dtype=np.uint8)
    for i in range(len(x)):
        label_map[:, i] = x[i]
    return label_map


def wave_map_norm(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    wave_top, _ = scipy.signal.find_peaks(x, prominence=(0.2,))
    if len(wave_top) != 0:
        ppg_with_top = np.zeros((len(x)))
        for i in range(len(wave_top)):
            if i == 0:
                ppg_with_top[0: wave_top[0]] = max_min_norm(x[0: wave_top[0]])
            if i > 0:
                ppg_with_top[wave_top[i - 1]: wave_top[i]] = max_min_norm(x[wave_top[i - 1]: wave_top[i]])
        if wave_top[len(wave_top) - 1] != len(ppg_with_top):
            ppg_with_top[wave_top[len(wave_top) - 1]:] = max_min_norm(x[wave_top[len(wave_top) - 1]:])
    elif len(wave_top) == 0:
        ppg_with_top = x
    return ppg_with_top


def compute_heart_rate(x, fps):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    wave_top, _ = scipy.signal.find_peaks(x, prominence=(0.2, 1))
    wave_spn = []
    for i in range(len(wave_top) - 1):
        x = wave_top[i + 1] - wave_top[i]
        wave_spn.append(x)
    mean_spn = np.mean(wave_spn)
    if len(wave_spn) == 0:
        mean_spn = 60 * fps / 80
    hr = int(60 / (mean_spn / fps))
    return hr


def rotate_90(x):
    x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
    return x


shift_num = 50
assert window_size % shift_num == 0
shift_drop_num = (window_size // shift_num) - 1


if raw_dataset_name == '01_ubfc':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    ubfc_fps = 30
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'ubfc_fps')
    sheet_out.write(1, 2, ubfc_fps)

    dataset_dir = './01_UBFC_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        node_map_path = dataset_dir + sub_0 + '/node_map.png'
        print(node_map_path)
        node_map = cv2.imread(node_map_path)

        gt_ppg = []
        label_xls_path = dataset_dir + sub_0 + '/0_gt_ppg_same_freq.xls'
        label_xls = xlrd.open_workbook_xls(label_xls_path)
        sheet = 'sheet'
        sheet_label = label_xls.sheet_by_name(sheet)
        rows_label = sheet_label.nrows
        for i in range(rows_label):
            x = float(sheet_label.cell(i, 0).value)
            gt_ppg.append(x)

        gt_hr = []
        hr_xls_path = dataset_dir + sub_0 + '/0_gt_hr_same_freq.xls'
        hr_xls = xlrd.open_workbook_xls(hr_xls_path)
        sheet_hr = hr_xls.sheet_by_name(sheet)
        rows_hr = sheet_hr.nrows
        for i in range(rows_hr):
            x = float(sheet_hr.cell(i, 0).value)
            gt_hr.append(x)

        assert len(gt_ppg) == node_map.shape[1]
        assert len(gt_ppg) == len(gt_hr)

        node_norm = node_map_norm(node_map)
        np.random.shuffle(node_map)
        node_shuffle = node_map_norm(node_map)

        gt_map = gen_wave_map(gt_ppg)
        gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

        k_mul = len(gt_ppg) // shift_num
        if k_mul > 1:
            for k in range(k_mul - shift_drop_num):
                gt_hr_seg = int(np.mean(gt_hr[k * shift_num: k * shift_num + shift_num]))
                compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=ubfc_fps)
                sheet_out.write(save_i + 1, 0, gt_hr_seg)
                sheet_out.write(save_i + 1, 1, compute_hr)
                save_i += 1

                node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                node_norm_seg = rgb2yuv(node_norm_seg)
                node_norm_seg = rotate_90(node_norm_seg)
                cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + str(k).zfill(3) + '_node_norm.png',
                            node_norm_seg)

                node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                node_shuffle_seg = rotate_90(node_shuffle_seg)
                cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + str(k).zfill(3) + '_node_shuffle.png',
                            node_shuffle_seg)

                gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                gt_map_seg = rotate_90(gt_map_seg)
                cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + str(k).zfill(3) + '_gt_map.png',
                            gt_map_seg)

                gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + str(k).zfill(3) + '_gt_norm.png',
                            gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '02_cohface':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    cohface_fps = 20
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'cohface_fps')
    sheet_out.write(1, 2, cohface_fps)

    dataset_dir = './02_COHFACE/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for light_0 in sub_list:
            node_map_path = sub_dir + light_0 + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            gt_ppg = []
            label_xls_path = sub_dir + light_0 + '/0_gt_pulse_same_freq.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = float(sheet_label.cell(i, 0).value)
                gt_ppg.append(x)

            assert len(gt_ppg) == node_map.shape[1]

            wave_top, _ = scipy.signal.find_peaks(gt_ppg)
            gt_ppg = gt_ppg[wave_top[0]:]
            node_map = node_map[:, wave_top[0]:]
            assert len(gt_ppg) == node_map.shape[1]

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            gt_map = gen_wave_map(gt_ppg)
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

            k_mul = len(gt_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=cohface_fps)
                    gt_hr_seg = compute_hr
                    sheet_out.write(save_i + 1, 0, gt_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + light_0 + '_' + str(k).zfill(3)
                                + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + light_0 + '_' + str(k).zfill(3)
                                + '_node_shuffle.png', node_shuffle_seg)

                    gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                    gt_map_seg = rotate_90(gt_map_seg)
                    cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + light_0 + '_' + str(k).zfill(3)
                                + '_gt_map.png', gt_map_seg)

                    gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                    gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                    cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + light_0 + '_' + str(k).zfill(3)
                                + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '03_vipl':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    vipl_fps = 30
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'vipl_fps')
    sheet_out.write(1, 2, vipl_fps)

    dataset_total = []
    dataset_xls = xlrd.open_workbook_xls('./03_VIPL_Dataset/VIPL_data_total_light.xls')
    sheet = 'sheet'
    sheet_data = dataset_xls.sheet_by_name(sheet)
    rows_data = sheet_data.nrows
    for j in range(rows_data):
        x = sheet_data.cell(j, 0).value
        x = './' + x[13:]
        dataset_total.append(x)

    for j in range(len(dataset_total)):
        node_map_path = dataset_total[j] + 'node_map.png'
        print(node_map_path)
        node_map = cv2.imread(node_map_path)

        gt_ppg = []
        label_xls_path = dataset_total[j] + '/0_gt_wave_same_freq.xls'
        label_xls = xlrd.open_workbook_xls(label_xls_path)
        sheet = 'sheet'
        sheet_label = label_xls.sheet_by_name(sheet)
        rows_label = sheet_label.nrows
        for i in range(rows_label):
            x = float(sheet_label.cell(i, 0).value)
            gt_ppg.append(x)

        gt_hr = []
        hr_xls_path = dataset_total[j] + '/0_gt_hr_same_freq.xls'
        hr_xls = xlrd.open_workbook_xls(hr_xls_path)
        sheet_hr = hr_xls.sheet_by_name(sheet)
        rows_hr = sheet_hr.nrows
        for i in range(rows_hr):
            x = float(sheet_hr.cell(i, 0).value)
            gt_hr.append(x)

        assert len(gt_ppg) == len(gt_hr)

        len_ = min(len(gt_ppg), node_map.shape[1])
        gt_ppg = gt_ppg[0: len_]
        gt_hr = gt_hr[0: len_]
        node_map = node_map[:, 0: len_]
        assert len(gt_ppg) == node_map.shape[1]
        assert len(gt_ppg) == len(gt_hr)

        node_norm = node_map_norm(node_map)
        np.random.shuffle(node_map)
        node_shuffle = node_map_norm(node_map)

        gt_map = gen_wave_map(gt_ppg)
        gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

        k_mul = len(gt_ppg) // shift_num
        if k_mul > 1:
            for k in range(k_mul - shift_drop_num):
                gt_hr_seg = int(np.mean(gt_hr[k * shift_num: k * shift_num + window_size]))
                compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=vipl_fps)
                sheet_out.write(save_i + 1, 0, gt_hr_seg)
                sheet_out.write(save_i + 1, 1, compute_hr)
                save_i += 1

                node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                node_norm_seg = rgb2yuv(node_norm_seg)
                node_norm_seg = rotate_90(node_norm_seg)
                cv2.imwrite(save_dir_sub_node_norm + dataset_total[j][28:35] + '_' + dataset_total[j][36:38] + '_s'
                            + dataset_total[j][45:46] + '_' + str(k).zfill(3) + '_node_norm.png', node_norm_seg)

                node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                node_shuffle_seg = rotate_90(node_shuffle_seg)
                cv2.imwrite(save_dir_sub_node_shuffle + dataset_total[j][28:35] + '_' + dataset_total[j][36:38] + '_s'
                            + dataset_total[j][45:46] + '_' + str(k).zfill(3) + '_node_shuffle.png', node_shuffle_seg)

                gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                gt_map_seg = rotate_90(gt_map_seg)
                cv2.imwrite(save_dir_sub_gt_map + dataset_total[j][28:35] + '_' + dataset_total[j][36:38] + '_s'
                            + dataset_total[j][45:46] + str(k).zfill(3) + '_gt_map.png', gt_map_seg)

                gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                cv2.imwrite(save_dir_sub_gt_norm + dataset_total[j][28:35] + '_' + dataset_total[j][36:38] + '_s'
                            + dataset_total[j][45:46] + '_' + str(k).zfill(3) + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '04_pure':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    pure_fps = 30
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'pure_fps')
    sheet_out.write(1, 2, pure_fps)

    dataset_dir = './04_PURE_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for v_0 in sub_list:
            node_map_path = sub_dir + v_0 + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            gt_ppg = []
            label_xls_path = sub_dir + v_0 + '/0_gt_bvp_same_freq.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = float(sheet_label.cell(i, 0).value)
                gt_ppg.append(x)

            gt_hr = []
            hr_xls_path = sub_dir + v_0 + '/0_gt_hr_same_freq.xls'
            hr_xls = xlrd.open_workbook_xls(hr_xls_path)
            sheet_hr = hr_xls.sheet_by_name(sheet)
            rows_hr = sheet_hr.nrows
            for i in range(rows_hr):
                x = float(sheet_hr.cell(i, 0).value)
                gt_hr.append(x)

            assert len(gt_ppg) == node_map.shape[1]
            assert len(gt_ppg) == len(gt_hr)

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            gt_map = gen_wave_map(gt_ppg)
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

            k_mul = len(gt_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    gt_hr_seg = int(np.mean(gt_hr[k * shift_num: k * shift_num + window_size]))
                    compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=pure_fps)
                    sheet_out.write(save_i + 1, 0, gt_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + v_0 + '_' + str(k).zfill(3)
                                + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + v_0 + '_' + str(k).zfill(3)
                                + '_node_shuffle.png', node_shuffle_seg)

                    gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                    gt_map_seg = rotate_90(gt_map_seg)
                    cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + v_0 + '_' + str(k).zfill(3)
                                + '_gt_map.png', gt_map_seg)

                    gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                    gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                    cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + v_0 + '_' + str(k).zfill(3)
                                + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '05_hci':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    hci_fps = 61
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'hci_fps')
    sheet_out.write(1, 2, hci_fps)

    dataset_dir = './05_HCI_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        node_map_path = dataset_dir + sub_0 + '/node_map.png'
        print(node_map_path)
        node_map = cv2.imread(node_map_path)

        gt_ppg_1 = []
        label_xls_path_1 = dataset_dir + sub_0 + '/0_gt_exg1_same_freq.xls'
        label_xls_1 = xlrd.open_workbook_xls(label_xls_path_1)
        sheet = 'sheet'
        sheet_label_1 = label_xls_1.sheet_by_name(sheet)
        rows_label_1 = sheet_label_1.nrows
        for i in range(rows_label_1):
            x = float(sheet_label_1.cell(i, 0).value)
            gt_ppg_1.append(x)

        gt_ppg_2 = []
        label_xls_path_2 = dataset_dir + sub_0 + '/0_gt_exg2_same_freq.xls'
        label_xls_2 = xlrd.open_workbook_xls(label_xls_path_2)
        sheet_label_2 = label_xls_2.sheet_by_name(sheet)
        rows_label_2 = sheet_label_2.nrows
        for i in range(rows_label_2):
            x = float(sheet_label_2.cell(i, 0).value)
            gt_ppg_2.append(x)

        gt_ppg_3 = []
        label_xls_path_3 = dataset_dir + sub_0 + '/0_gt_exg3_same_freq.xls'
        label_xls_3 = xlrd.open_workbook_xls(label_xls_path_3)
        sheet_label_3 = label_xls_3.sheet_by_name(sheet)
        rows_label_3 = sheet_label_3.nrows
        for i in range(rows_label_3):
            x = float(sheet_label_3.cell(i, 0).value)
            gt_ppg_3.append(x)

        gt_hr = []
        hr_xls_path = dataset_dir + sub_0 + '/0_gt_hr_same_freq.xls'
        hr_xls = xlrd.open_workbook_xls(hr_xls_path)
        sheet_hr = hr_xls.sheet_by_name(sheet)
        rows_hr = sheet_hr.nrows
        for i in range(rows_hr):
            x = float(sheet_hr.cell(i, 0).value)
            gt_hr.append(x)

        assert len(gt_ppg_1) == node_map.shape[1]
        assert len(gt_ppg_1) == len(gt_ppg_2)
        assert len(gt_ppg_1) == len(gt_ppg_3)
        assert len(gt_ppg_1) == len(gt_hr)

        node_norm = node_map_norm(node_map)
        np.random.shuffle(node_map)
        node_shuffle_1 = node_map_norm(node_map)
        np.random.shuffle(node_map)
        node_shuffle_2 = node_map_norm(node_map)
        np.random.shuffle(node_map)
        node_shuffle_3 = node_map_norm(node_map)

        gt_map_1 = gen_wave_map(gt_ppg_1)

        ecg_top_1, _ = scipy.signal.find_peaks(gt_ppg_1)
        if len(ecg_top_1) != 0:
            gt_norm_map_1 = gen_wave_map(wave_map_norm(gt_ppg_1))
        else:
            gt_norm_map_1 = gen_wave_map(gt_ppg_1)

        gt_map_2 = gen_wave_map(gt_ppg_2)
        ecg_top_2, _ = scipy.signal.find_peaks(gt_ppg_2)
        if len(ecg_top_2) != 0:
            gt_norm_map_2 = gen_wave_map(wave_map_norm(gt_ppg_2))
        else:
            gt_norm_map_2 = gen_wave_map(gt_ppg_2)

        gt_map_3 = gen_wave_map(gt_ppg_3)
        ecg_top_3, _ = scipy.signal.find_peaks(gt_ppg_3)
        if len(ecg_top_3) != 0:
            gt_norm_map_3 = gen_wave_map(wave_map_norm(gt_ppg_3))
        else:
            gt_norm_map_3 = gen_wave_map(gt_ppg_3)

        k_mul = len(gt_ppg_1) // shift_num
        if k_mul > 1:
            for k in range(k_mul - shift_drop_num):
                gt_hr_seg = int(np.mean(gt_hr[k * shift_num: k * shift_num + window_size]))
                sheet_out.write(save_i + 1, 0, gt_hr_seg)
                sheet_out.write(save_i + 2, 0, gt_hr_seg)
                sheet_out.write(save_i + 3, 0, gt_hr_seg)
                sheet_out.write(save_i + 1, 1, gt_hr_seg)
                sheet_out.write(save_i + 2, 1, gt_hr_seg)
                sheet_out.write(save_i + 3, 1, gt_hr_seg)
                save_i += 3

                node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                node_norm_seg = rgb2yuv(node_norm_seg)
                node_norm_seg = rotate_90(node_norm_seg)
                cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + str(k).zfill(3) + '_node_norm_1.png',
                            node_norm_seg)
                cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + str(k).zfill(3) + '_node_norm_2.png',
                            node_norm_seg)
                cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + str(k).zfill(3) + '_node_norm_3.png',
                            node_norm_seg)

                node_shuffle_seg_1 = node_map_norm(node_shuffle_1[:, k * shift_num: k * shift_num + window_size])
                node_shuffle_seg_1 = rgb2yuv(node_shuffle_seg_1)
                node_shuffle_seg_1 = rotate_90(node_shuffle_seg_1)
                cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + str(k).zfill(3) + '_node_shuffle_1.png',
                            node_shuffle_seg_1)

                node_shuffle_seg_2 = node_map_norm(node_shuffle_2[:, k * shift_num: k * shift_num + window_size])
                node_shuffle_seg_2 = rgb2yuv(node_shuffle_seg_2)
                node_shuffle_seg_2 = rotate_90(node_shuffle_seg_2)
                cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + str(k).zfill(3) + '_node_shuffle_2.png',
                            node_shuffle_seg_2)

                node_shuffle_seg_3 = node_map_norm(node_shuffle_3[:, k * shift_num: k * shift_num + window_size])
                node_shuffle_seg_3 = rgb2yuv(node_shuffle_seg_3)
                node_shuffle_seg_3 = rotate_90(node_shuffle_seg_3)
                cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + str(k).zfill(3) + '_node_shuffle_3.png',
                            node_shuffle_seg_3)

                gt_map_seg_1 = node_map_norm(gt_map_1[:, k * shift_num: k * shift_num + window_size])
                gt_map_seg_1 = rotate_90(gt_map_seg_1)
                cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + str(k).zfill(3) + '_gt_map_1.png',
                            gt_map_seg_1)

                gt_norm_map_seg_1 = node_map_norm(gt_norm_map_1[:, k * shift_num: k * shift_num + window_size])
                gt_norm_map_seg_1 = rotate_90(gt_norm_map_seg_1)
                cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + str(k).zfill(3) + '_gt_norm_1.png',
                            gt_norm_map_seg_1)

                gt_map_seg_2 = node_map_norm(gt_map_2[:, k * shift_num: k * shift_num + window_size])
                gt_map_seg_2 = rotate_90(gt_map_seg_2)
                cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + str(k).zfill(3) + '_gt_map_2.png',
                            gt_map_seg_2)

                gt_norm_map_seg_2 = node_map_norm(gt_norm_map_2[:, k * shift_num: k * shift_num + window_size])
                gt_norm_map_seg_2 = rotate_90(gt_norm_map_seg_2)
                cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + str(k).zfill(3) + '_gt_norm_2.png',
                            gt_norm_map_seg_2)

                gt_map_seg_3 = node_map_norm(gt_map_3[:, k * shift_num: k * shift_num + window_size])
                gt_map_seg_3 = rotate_90(gt_map_seg_3)
                cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + str(k).zfill(3) + '_gt_map_3.png',
                            gt_map_seg_3)

                gt_norm_map_seg_3 = node_map_norm(gt_norm_map_3[:, k * shift_num: k * shift_num + window_size])
                gt_norm_map_seg_3 = rotate_90(gt_norm_map_seg_3)
                cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + str(k).zfill(3) + '_gt_norm_3.png',
                            gt_norm_map_seg_3)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '05_hci_ppg_label':
    shift_num = 75
    assert window_size % shift_num == 0
    shift_drop_num = (window_size // shift_num) - 1

    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    hci_fps = 61
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'hci_fps')
    sheet_out.write(1, 2, hci_fps)

    dataset_dir = './05_HCI_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()

    for sub_0 in dataset_list:
        node_map_path = dataset_dir + sub_0 + '/node_map.png'
        print(node_map_path)
        node_map = cv2.imread(node_map_path)

        gt_ecg = []
        label_xls_path = dataset_dir + sub_0 + '/0_gt_exg3_same_freq.xls'
        label_xls = xlrd.open_workbook_xls(label_xls_path)
        sheet = 'sheet'
        sheet_label = label_xls.sheet_by_name(sheet)
        rows_label = sheet_label.nrows
        for i in range(rows_label):
            x = float(sheet_label.cell(i, 0).value)
            gt_ecg.append(x)

        gt_ecg = (gt_ecg - np.min(gt_ecg)) / (np.max(gt_ecg) - np.min(gt_ecg))
        wave_top, _ = scipy.signal.find_peaks(gt_ecg, prominence=(0.3,))
        if len(wave_top) > 2:
            gt_ppg = generative_smooth_bvp(gt_ecg, line_top=wave_top)
        else:
            gt_ppg = gt_ecg

        gt_hr = []
        hr_xls_path = dataset_dir + sub_0 + '/0_gt_hr_same_freq.xls'
        hr_xls = xlrd.open_workbook_xls(hr_xls_path)
        sheet_hr = hr_xls.sheet_by_name(sheet)
        rows_hr = sheet_hr.nrows
        for i in range(rows_hr):
            x = float(sheet_hr.cell(i, 0).value)
            gt_hr.append(x)

        assert len(gt_ppg) == node_map.shape[1]
        assert len(gt_ppg) == len(gt_hr)

        node_norm = node_map_norm(node_map)
        np.random.shuffle(node_map)
        node_shuffle_1 = node_map_norm(node_map)
        np.random.shuffle(node_map)
        node_shuffle_2 = node_map_norm(node_map)
        np.random.shuffle(node_map)
        node_shuffle_3 = node_map_norm(node_map)

        gt_map = gen_wave_map(gt_ppg)

        ecg_top, _ = scipy.signal.find_peaks(gt_ppg)
        if len(ecg_top) != 0:
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))
        else:
            gt_norm_map = gen_wave_map(gt_ppg)

        k_mul = len(gt_ppg) // shift_num
        if k_mul > 1:
            for k in range(k_mul - shift_drop_num):
                gt_hr_seg = int(np.mean(gt_hr[k * shift_num: k * shift_num + window_size]))
                sheet_out.write(save_i + 1, 0, gt_hr_seg)
                sheet_out.write(save_i + 1, 1, gt_hr_seg)
                save_i += 1

                node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                node_norm_seg = rgb2yuv(node_norm_seg)
                node_norm_seg = rotate_90(node_norm_seg)
                cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + str(k).zfill(3) + '_node_norm.png',
                            node_norm_seg)

                node_shuffle_seg = node_map_norm(node_shuffle_1[:, k * shift_num: k * shift_num + window_size])
                node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                node_shuffle_seg = rotate_90(node_shuffle_seg)
                cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + str(k).zfill(3) + '_node_shuffle.png',
                            node_shuffle_seg)

                gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                gt_map_seg = rotate_90(gt_map_seg)
                cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + str(k).zfill(3) + '_gt_map.png', gt_map_seg)

                gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + str(k).zfill(3) + '_gt_norm.png', gt_norm_map_seg)

    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '06_nirp_indoor':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    nirp_fps = 30
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'indoor_fps')
    sheet_out.write(1, 2, nirp_fps)

    dataset_dir = './06_NIRP_Indoor_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for motion_0 in sub_list:
            node_map_path = sub_dir + motion_0 + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            gt_ppg = []
            label_xls_path = sub_dir + motion_0 + '/0_gt_ppg_same_freq.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = float(sheet_label.cell(i, 0).value)
                gt_ppg.append(x)

            len_ = min(len(gt_ppg), node_map.shape[1])
            gt_ppg = gt_ppg[0: len_]
            node_map = node_map[:, 0: len_]
            assert len(gt_ppg) == node_map.shape[1]

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            gt_map = gen_wave_map(gt_ppg)
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

            k_mul = len(gt_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=nirp_fps)
                    gt_hr_seg = compute_hr
                    sheet_out.write(save_i + 1, 0, gt_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + motion_0[0: 1] + '_' + str(k).zfill(3)
                                + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + motion_0[0: 1] + '_' + str(k).zfill(3)
                                + '_node_shuffle.png', node_shuffle_seg)

                    gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                    gt_map_seg = rotate_90(gt_map_seg)
                    cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + motion_0[0: 1] + '_' + str(k).zfill(3)
                                + '_gt_map.png', gt_map_seg)

                    gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                    gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                    cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + motion_0[0: 1] + '_' + str(k).zfill(3)
                                + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '07_nirp_car940':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    nirp_fps = 30
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'car940_fps')
    sheet_out.write(1, 2, nirp_fps)

    dataset_dir = './07_NIRP_Car940_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for motion_0 in sub_list:
            if motion_0 == 'driving_small_motion_940':
                m_0 = 'dm'
            elif motion_0 == 'driving_still_940':
                m_0 = 'ds'
            elif motion_0 == 'garage_small_motion_940':
                m_0 = 'gm'
            elif motion_0 == 'garage_still_940':
                m_0 = 'gs'
            node_map_path = sub_dir + motion_0 + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            gt_ppg = []
            label_xls_path = sub_dir + motion_0 + '/0_gt_ppg_same_freq.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = float(sheet_label.cell(i, 0).value)
                gt_ppg.append(x)

            len_ = min(len(gt_ppg), node_map.shape[1])
            gt_ppg = gt_ppg[0: len_]
            node_map = node_map[:, 0: len_]
            assert len(gt_ppg) == node_map.shape[1]

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            gt_map = gen_wave_map(gt_ppg)
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

            k_mul = len(gt_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=nirp_fps)
                    gt_hr_seg = compute_hr
                    sheet_out.write(save_i + 1, 0, gt_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + m_0 + '_' + str(k).zfill(3)
                                + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + m_0 + '_' + str(k).zfill(3)
                                + '_node_shuffle.png', node_shuffle_seg)

                    gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                    gt_map_seg = rotate_90(gt_map_seg)
                    cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + m_0 + '_' + str(k).zfill(3)
                                + '_gt_map.png', gt_map_seg)

                    gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                    gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                    cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + m_0 + '_' + str(k).zfill(3)
                                + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '08_nirp_car975':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    nirp_fps = 30
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'car975_fps')
    sheet_out.write(1, 2, nirp_fps)

    dataset_dir = './08_NIRP_Car975_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()

    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for motion_0 in sub_list:
            if motion_0 == 'Driving_large_motion_975':
                m_0 = 'dlm'
            elif motion_0 == 'Driving_small_motion_975':
                m_0 = 'dsm'
            elif motion_0 == 'Driving_still_975':
                m_0 = 'dst'
            elif motion_0 == 'Garage_large_motion_975':
                m_0 = 'glm'
            elif motion_0 == 'Garage_small_motion_975':
                m_0 = 'gsm'
            elif motion_0 == 'Garage_still_975':
                m_0 = 'gst'
            node_map_path = sub_dir + motion_0 + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            gt_ppg = []
            label_xls_path = sub_dir + motion_0 + '/0_gt_ppg_same_freq.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = float(sheet_label.cell(i, 0).value)
                gt_ppg.append(x)

            len_ = min(len(gt_ppg), node_map.shape[1])
            gt_ppg = gt_ppg[0: len_]
            node_map = node_map[:, 0: len_]
            assert len(gt_ppg) == node_map.shape[1]

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            gt_map = gen_wave_map(gt_ppg)
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

            k_mul = len(gt_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=nirp_fps)
                    gt_hr_seg = compute_hr
                    sheet_out.write(save_i + 1, 0, gt_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + m_0 + '_' + str(k).zfill(3)
                                + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + m_0 + '_' + str(k).zfill(3)
                                + '_node_shuffle.png', node_shuffle_seg)

                    gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                    if np.max(gt_map_seg) != np.min(gt_map_seg):
                        gt_map_seg_last = gt_map_seg
                    else:
                        gt_map_seg = gt_map_seg_last
                    gt_map_seg = rotate_90(gt_map_seg)
                    cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + m_0 + '_' + str(k).zfill(3)
                                + '_gt_map.png', gt_map_seg)

                    gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                    if np.max(gt_norm_map_seg) != np.min(gt_norm_map_seg):
                        gt_norm_map_seg_last = gt_norm_map_seg
                    else:
                        gt_norm_map_seg = gt_norm_map_seg_last
                    gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                    cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + m_0 + '_' + str(k).zfill(3)
                                + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '09_buaa':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    buaa_fps = 30
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'buaa_fps')
    sheet_out.write(1, 2, buaa_fps)

    dataset_dir = './09_BUAA_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for lux_0 in range(len(sub_list)):
            node_map_path = sub_dir + sub_list[lux_0] + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            gt_ppg = []
            label_xls_path = sub_dir + sub_list[lux_0] + '/0_gt_wave_same_freq.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = float(sheet_label.cell(i, 0).value)
                gt_ppg.append(x)

            gt_hr = []
            hr_xls_path = sub_dir + sub_list[lux_0] + '/0_gt_hr_same_freq.xls'
            hr_xls = xlrd.open_workbook_xls(hr_xls_path)
            sheet_hr = hr_xls.sheet_by_name(sheet)
            rows_hr = sheet_hr.nrows
            for i in range(rows_hr):
                x = float(sheet_hr.cell(i, 0).value)
                gt_hr.append(x)

            assert len(gt_ppg) == node_map.shape[1]
            assert len(gt_ppg) == len(gt_hr)

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            gt_map = gen_wave_map(gt_ppg)
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

            k_mul = len(gt_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    gt_hr_seg = int(np.mean(gt_hr[k * shift_num: k * shift_num + window_size]))
                    compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=buaa_fps)
                    sheet_out.write(save_i + 1, 0, gt_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_lux' + str(lux_0).zfill(2) + '_'
                                + str(k).zfill(3) + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_lux' + str(lux_0).zfill(2) + '_'
                                + str(k).zfill(3) + '_node_shuffle.png', node_shuffle_seg)

                    gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                    gt_map_seg = rotate_90(gt_map_seg)
                    cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_lux' + str(lux_0).zfill(2) + '_'
                                + str(k).zfill(3) + '_gt_map.png', gt_map_seg)

                    gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                    gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                    cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_lux' + str(lux_0).zfill(2) + '_'
                                + str(k).zfill(3) + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '10_Tokyo':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    tokyo_fps = 30
    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'tokyo_fps')
    sheet_out.write(1, 2, tokyo_fps)

    dataset_dir = './10_Tokyo_Dataset/Main_data/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/30fps/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for seg_0 in range(len(sub_list)):
            node_map_path = sub_dir + sub_list[seg_0] + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            gt_ppg = []
            label_xls_path = sub_dir + sub_list[seg_0] + '/0_gt_ppg_same_freq.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = float(sheet_label.cell(i, 0).value)
                gt_ppg.append(x)

            assert len(gt_ppg) == node_map.shape[1]

            wave_top, _ = scipy.signal.find_peaks(gt_ppg)
            gt_ppg = gt_ppg[wave_top[0]:]
            node_map = node_map[:, wave_top[0]:]
            assert len(gt_ppg) == node_map.shape[1]

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            gt_map = gen_wave_map(gt_ppg)
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

            k_mul = len(gt_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=tokyo_fps)
                    gt_hr_seg = compute_hr
                    sheet_out.write(save_i + 1, 0, gt_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_seg' + str(seg_0) + '_' + str(k).zfill(3)
                                + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_seg' + str(seg_0) + '_' + str(k).zfill(3)
                                + '_node_shuffle.png', node_shuffle_seg)

                    gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                    gt_map_seg = rotate_90(gt_map_seg)
                    cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_seg' + str(seg_0) + '_' + str(k).zfill(3)
                                + '_gt_map.png', gt_map_seg)

                    gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                    gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                    cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_seg' + str(seg_0) + '_' + str(k).zfill(3)
                                + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')


elif raw_dataset_name == '11_VIPL_V2':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_gt_norm = save_dir + 'gt_norm/'
    if not os.path.exists(save_dir_sub_gt_norm):
        os.makedirs(save_dir_sub_gt_norm)

    save_dir_sub_gt_map = save_dir + 'gt_map/'
    if not os.path.exists(save_dir_sub_gt_map):
        os.makedirs(save_dir_sub_gt_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'gt_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'fps')
    sheet_out.write(1, 2, '30')

    dataset_dir = './11_VIPL_V2/VV2_Main/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()

        for video_0 in sub_list:
            the_dir = sub_dir + video_0
            if the_dir.endswith('.db'):
                os.remove(the_dir)

        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for video_0 in sub_list:
            node_map_path = sub_dir + video_0 + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            gt_ppg = []
            label_xls_path = sub_dir + video_0 + '/gt_label.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = sheet_label.cell(i, 0).value
                gt_ppg.append(x)
            gt_ppg = gt_ppg[1:]

            gt_hr = float(sheet_label.cell(1, 1).value)
            vipl_v2_fps_one = float(sheet_label.cell(1, 3).value)
            vipl_v2_fps_one = 30

            assert len(gt_ppg) == node_map.shape[1]

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            gt_map = gen_wave_map(gt_ppg)
            gt_norm_map = gen_wave_map(wave_map_norm(gt_ppg))

            k_mul = len(gt_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    compute_hr = compute_heart_rate(gt_ppg[k * shift_num: k * shift_num + window_size], fps=vipl_v2_fps_one)
                    gt_hr_seg = gt_hr
                    sheet_out.write(save_i + 1, 0, gt_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + video_0 + '_' + str(k).zfill(3)
                                + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + video_0 + '_' + str(k).zfill(3)
                                + '_node_shuffle.png', node_shuffle_seg)

                    gt_map_seg = node_map_norm(gt_map[:, k * shift_num: k * shift_num + window_size])
                    gt_map_seg = rotate_90(gt_map_seg)
                    cv2.imwrite(save_dir_sub_gt_map + sub_0 + '_' + video_0 + '_' + str(k).zfill(3)
                                + '_gt_map.png', gt_map_seg)

                    gt_norm_map_seg = node_map_norm(gt_norm_map[:, k * shift_num: k * shift_num + window_size])
                    gt_norm_map_seg = rotate_90(gt_norm_map_seg)
                    cv2.imwrite(save_dir_sub_gt_norm + sub_0 + '_' + video_0 + '_' + str(k).zfill(3)
                                + '_gt_norm.png', gt_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')
