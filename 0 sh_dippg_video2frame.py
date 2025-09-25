import cv2
import os

"""
01_ubfc, 02_cohface, 03_vipl, 04_pure, 05_hci,
06_nirp_indoor, 07_nirp_car940, 08_nirp_car975,
09_buaa, 10_tokyo, 11_vv2, 13_mmpd, 14_yawdd
"""

data_name = '01_ubfc'

if data_name == '01_ubfc':
    VideoData = []
    VideoRoot = '../rppg_data/01_UBFC_Dataset/DATASET/'
    dir_list_0 = os.listdir(VideoRoot)
    dir_list_0.sort()
    for dir_1 in dir_list_0:
        dir_root_1 = VideoRoot + dir_1 + '/'
        VideoData.append(dir_root_1)

    for i in range(len(VideoData)):
        print(VideoData[i])
        VideoSingle = VideoData[i] + 'vid.avi'
        OutImg = VideoData[i] + 'frame/'
        if not os.path.exists(OutImg):
            os.makedirs(OutImg)
        vc = cv2.VideoCapture(VideoSingle)
        c = 0
        rval = vc.isOpened()
        while rval:
            c = c + 1
            rval, frame = vc.read()
            if rval:
                cv2.imwrite(OutImg + 'Frame_' + str(c).zfill(4) + '.jpg', frame)
                cv2.waitKey(1)
            else:
                break
        vc.release()


elif data_name == '02_cohface':
    VideoData = []
    VideoRoot = '../rppg_data/02_COHFACE/cohface/'
    dir_list_0 = os.listdir(VideoRoot)
    dir_list_0.sort()
    for dir_1 in dir_list_0:
        dir_root_1 = VideoRoot + dir_1 + '/'
        dir_list_1 = os.listdir(dir_root_1)
        dir_list_1.sort()
        for dir_2 in dir_list_1:
            dir_root_2 = dir_root_1 + dir_2 + '/'
            VideoData.append(dir_root_2)

    for i in range(len(VideoData)):
        print(VideoData[i])
        VideoSingle = VideoData[i] + 'data.avi'
        OutImg = VideoData[i] + 'frame/'
        if not os.path.exists(OutImg):
            os.makedirs(OutImg)
        vc = cv2.VideoCapture(VideoSingle)
        c = 0
        rval = vc.isOpened()
        while rval:
            c = c + 1
            rval, frame = vc.read()
            if rval:
                cv2.imwrite(OutImg + 'Frame_' + str(c).zfill(4) + '.jpg', frame)
                cv2.waitKey(1)
            else:
                break
        vc.release()


elif data_name == '03_vipl':
    data_xls = xlrd.open_workbook_xls('../rppg_data/03_VIPL_Dataset/vipl_p0_light.xls')
    sheet = 'sheet'
    sheet_data = data_xls.sheet_by_name(sheet)
    rows_data = sheet_data.nrows
    for rows_i in range(rows_data):
        dir_root = '.' + sheet_data.cell(rows_i, 1).value
        VideoData.append(dir_root)

    for i in range(len(VideoData)):
        print(VideoData[i])
        VideoSingle = VideoData[i] + 'video.avi'
        OutImg = VideoData[i] + 'frame/'
        if not os.path.exists(OutImg):
            os.makedirs(OutImg)
        vc = cv2.VideoCapture(VideoSingle)
        c = 0
        rval = vc.isOpened()
        while rval:
            c = c + 1
            rval, frame = vc.read()
            if rval:
                cv2.imwrite(OutImg + 'Frame_' + str(c).zfill(4) + '.jpg', frame)
                cv2.waitKey(1)
            else:
                break
        vc.release()


# the PURE, MR-NIRP (Indoor, 940, 975), BUAA are frames.
