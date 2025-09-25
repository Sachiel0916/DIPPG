import os
import cv2
import numpy as np
from scipy.sparse import spdiags, csr_matrix
# from pypardiso import spsolve
from scipy.sparse.linalg import spsolve


def tsmooth(I, lambda_=0.01, sigma=5.0, sharpness=0.02, maxIter=4):
    I = I.astype(np.float32) / 255.0
    x = I.copy()
    sigma_iter = sigma
    lambda_ /= 2.0
    dec = 2.0

    for _ in range(maxIter):
        wx, wy = computeTextureWeights(x, sigma_iter, sharpness)
        x = solveLinearEquation(I, wx, wy, lambda_)
        sigma_iter = max(sigma_iter / dec, 0.5)

    return (x * 255).astype(np.uint8)


def computeTextureWeights(fin, sigma, sharpness):
    fx = np.diff(fin, axis=1)
    fx = np.pad(fx, ((0, 0), (0, 1), (0, 0)), mode="constant")
    fy = np.diff(fin, axis=0)
    fy = np.pad(fy, ((0, 1), (0, 0), (0, 0)), mode="constant")

    vareps_s = sharpness
    vareps = 0.001

    wto = (
        np.maximum(np.sum(np.sqrt(fx**2 + fy**2), axis=2) / fin.shape[2], vareps_s)
        ** -1
    )
    fbin = lpfilter(fin, sigma)
    gfx = np.diff(fbin, axis=1)
    gfx = np.pad(gfx, ((0, 0), (0, 1), (0, 0)), mode="constant")
    gfy = np.diff(fbin, axis=0)
    gfy = np.pad(gfy, ((0, 1), (0, 0), (0, 0)), mode="constant")

    wtbx = np.maximum(np.sum(np.abs(gfx), axis=2) / fin.shape[2], vareps) ** -1
    wtby = np.maximum(np.sum(np.abs(gfy), axis=2) / fin.shape[2], vareps) ** -1

    retx = wtbx * wto
    rety = wtby * wto

    retx[:, -1] = 0
    rety[-1, :] = 0

    return retx, rety


def conv2_sep(im, sigma):
    ksize = max(round(5 * sigma), 1)
    if ksize % 2 == 0:
        ksize += 1
    g = cv2.getGaussianKernel(ksize, sigma)
    ret = cv2.filter2D(im, -1, g)
    ret = cv2.filter2D(ret, -1, g.T)
    return ret


def lpfilter(FImg, sigma):
    FBImg = np.zeros_like(FImg)
    for ic in range(FImg.shape[2]):
        FBImg[:, :, ic] = conv2_sep(FImg[:, :, ic], sigma)
    return FBImg


def solveLinearEquation(IN, wx, wy, lambda_):
    r, c, ch = IN.shape
    k = r * c

    dx = -lambda_ * wx.ravel(order="F")
    dy = -lambda_ * wy.ravel(order="F")

    B = np.vstack((dx, dy))
    d = [-r, -1]
    A = spdiags(B, d, k, k)

    e = dx
    w = np.pad(dx[:-r], (r, 0), "constant")
    s = dy
    n = np.pad(dy[:-1], (1, 0), "constant")
    D = 1 - (e + w + s + n)
    A = A + A.T + spdiags(D, 0, k, k)

    A = csr_matrix(A)

    OUT = np.zeros_like(IN)
    for i in range(ch):
        tin = IN[:, :, i].ravel(order="F")
        tout = spsolve(A.astype(np.float64), tin.astype(np.float64))
        OUT[:, :, i] = tout.reshape((r, c), order="F")

    return OUT


data_dir = ['01_ubfc', '02_cohface', '03_vipl', '04_pure',
            '06_nirp_indoor', '07_nirp_car940', '08_nirp_car975',
            '09_buaa', '10_tokyo', '05_hci']

for k in range(len(data_dir)):
    root_dir = '../traffic_dataset/' + data_dir[k] + '/'

    img_dir = ['norm_face', 'norm_glob', 'norm_without_face',
               'wo_norm_face', 'wo_norm_glob', 'wo_norm_without_face',
               'yuv_face', 'yuv_glob', 'yuv_without_face']

    for j in range(len(img_dir)):
        input_dir = root_dir + img_dir[j] + '/'
        img_list = os.listdir(input_dir + '/')
        img_list.sort()

        img_save_dir = root_dir + img_dir[j] + '_rtv/'
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        i = 0
        for img_0 in img_list:
            print(input_dir, i)
            i += 1
            img_path = input_dir + img_0
            if img_path.endswith('.png'):
                img = cv2.imread(img_path)
                s = tsmooth(img, maxIter=2)
                cv2.imwrite(img_save_dir + img_0, s)
