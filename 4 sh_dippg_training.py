import os
import tqdm
import xlrd
import xlwt
import random
from random import randint
import argparse
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from typing import TypeVar, Generic
T_co = TypeVar('T_co', covariant=True)

from torchvision import transforms

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from sh_model.CNN_DeepPhys import DeepPhys          # Deepphys: Video-based physiological measurement using convolutional attention networks. (ECCV'18)
from sh_model.CNN_TS_CAN import TS_CAN              # Multi-task temporal shift attention networks for on-device contactless vitals measurement. (NIPS'20)
from sh_model.CNN_DualGAN import DualGAN            # Dual-gan: Joint bvp and noise modeling for remote physiological measurement. (CVPR'21)
from sh_model.CNN_PFE_TFA import PFE_TFA            # Learning motion-robust remote photoplethysmography through arbitrary resolution videos. (AAAI'23)
from sh_model.CNN_NEST import NEST                  # Neuron structure modeling for generalizable remote physiological measurement. (CVPR'23)
from sh_model.CNN_ND_DeepRPPG import ND_DeepRPPG    # Robust remote photoplethysmography estimation with environmental noise disentanglement. (IEEE TIP'24)
from sh_model.CNN_rPPG_HiBa import rPPG_HiBa        # Rppg-hiba: Hierarchical balanced framework for remote physiological measurement. (ACM MM'24)
from sh_model.CNN_DD_rPPGNet import DD_rPPGNet      # Dd-rppgnet: De-interfering and descriptive feature learning for unsupervised rppg estimation. (IEEE TIFS'25)
from sh_model.TIFS2025_DD_rPPGNet_main.loss.contrastiveLoss import ContrastLoss
from sh_model.TIFS2025_DD_rPPGNet_main.loss.FrequencyContrast import FrequencyContrast
from sh_model.TIFS2025_DD_rPPGNet_main.loss.KMeans import KMeans

from sh_model.ViT_PhysFormer import PhysFormer                      # Physformer: Facial video-based physiological measurement with temporal difference transformer. (CVPR'22)
from sh_model.ViT_EfficientPhys import EfficientPhys_Transformer    # Efficientphys: Enabling simple, fast and accurate camera-based cardiac measurement. (WACV'23)
from sh_model.ViT_NDNet import SwinUNet as NDNet                    # Remote photoplethysmography in real-world and extreme lighting scenarios. (CVPR'25)
from sh_model.ViT_RhythmFormer import RhythmFormer                  # Rhythmformer: Extracting patterned rppg signals based on periodic sparse attention. (PR'25)

from sh_model.Mamba_physmamba_td import PhysMambaTD         # Physmamba: Efficient remote physiological measurement with slowfast temporal difference mamba. (CCBR'24)
from sh_model.Mamba_rhythmmamba import RhythmMamba          # Rhythmmamba: Fast, lightweight, and accurate remote physiological measurement. (AAAI'25)
from sh_model.Mamba_physmamba_cassd import PhysMambaCASSD   # Physmamba: Synergistic state space duality model for remote physiological measurement. (ICANN'25)

from sh_model.Our_DIPPG import DIPPG as Our_Net
from sh_model.Our_DIPPG_loss import Our_FCL as FCL

from sh_model.utils import NegPearson


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='02_cohface',
                    help='01_ubfc, 02_cohface, 03_vipl, 04_pure, 05_hci, 06_nirp_indoor, 07_nirp_car940, 08_nirp_car975, 09_buaa, 10_tokyo')
parser.add_argument('--input_img_pattern', default='yuv', help='rgb, yuv')
parser.add_argument('--limit_train_scale', default=False)
parser.add_argument('--whether_rtv', default=True)
parser.add_argument('--train_data_scale', default=1000)
parser.add_argument('--batch_size', default=4)
parser.add_argument('--data_shuffle', default=False)
parser.add_argument('--epochs', default=100)
parser.add_argument('--device', default='cuda')
parser.add_argument('--temporal_size', default=320)
parser.add_argument('--img_size', default=64)
parser.add_argument('--network_name', default='Our_Net',
                    help='DeepPhys, DualGAN, TS_CAN, PFE_TFA, NEST, ND_DeepRPPG, rPPG_HiBa, DD_rPPGNet, NDNet'
                         'EfficientPhys, PhysFormer, RhythmFormer,'
                         'RhythmMamba, PhysMambaTD, PhysMambaCASSD,'
                         'Our_Net')
parser.add_argument('--save_weight_name', default='.pt')
parser.add_argument('--pretraining', default=False)
parser.add_argument('--learning_rate', default=1e-5)
parser.add_argument('--save_weight', default=True)
parser.add_argument('--save_train_log', default=True)
args = parser.parse_args()

print('----------', args.network_name, '----------')

input_img_pattern = args.input_img_pattern
temporal_size = args.temporal_size
img_size = args.img_size
batch_size = args.batch_size


save_dir = './training_save_dir/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_dir = './training_save_dir/' + args.network_name + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


video_fps = 25


def img_to_device(x, c):
    x = x.to(args.device)
    x = x.view(batch_size, c, temporal_size, img_size)
    x = (x - torch.mean(x)) / torch.std(x)
    return x


def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)
        signal = np.pad(signal, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant')
    freqs = np.fft.fftfreq(len(signal), 1 / Fs) * 60
    ps = np.abs(np.fft.fft(signal))**2
    cutoff = len(freqs)//2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps


class DataAugmentation(object):
    def __init__(self):
        super(DataAugmentation, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((temporal_size, img_size))
        ])

    def __call__(self, x):
        x = self.transform(x)
        return x


def get_random_bvp(xs):
    r1 = randint(0, len(xs) - 1)
    r2 = randint(0, len(xs) - 1)
    r3 = randint(0, len(xs) - 1)
    r4 = randint(0, len(xs) - 1)

    bvp_map1 = xs[r1]
    bvp_map2 = xs[r2]
    bvp_map3 = xs[r3]
    bvp_map4 = xs[r4]

    bvp1 = Image.open(bvp_map1)
    bvp2 = Image.open(bvp_map2)
    bvp3 = Image.open(bvp_map3)
    bvp4 = Image.open(bvp_map4)

    bvp1 = bvp1.resize((1, temporal_size))
    bvp2 = bvp2.resize((1, temporal_size))
    bvp3 = bvp3.resize((1, temporal_size))
    bvp4 = bvp4.resize((1, temporal_size))

    bvp1 = np.array(bvp1)
    bvp2 = np.array(bvp2)
    bvp3 = np.array(bvp3)
    bvp4 = np.array(bvp4)

    out = np.concatenate([bvp1, bvp2, bvp3, bvp4], axis=1)
    return out


class GenDataset(data.Dataset):
    def __init__(self, img_faces, img_globals, img_backs, face_rtvs, gt_bvps, trans):
        super(GenDataset, self).__init__()
        self.img_faces = img_faces
        self.img_globals = img_globals
        self.img_backs = img_backs
        self.face_rtvs = face_rtvs

        self.gt_bvps = gt_bvps

        self.trans = trans
        self.data_len = len(img_faces)

    def __getitem__(self, index) -> T_co:
        img_face = self.img_faces[index]
        img_global = self.img_globals[index]
        img_back = self.img_backs[index]

        face_rtv = self.face_rtvs[index]
        gt_bvp = self.gt_bvps[index]

        img_face = Image.open(img_face)
        img_face = img_face.convert('RGB')
        face_tensor = self.trans(img_face)

        img_global = Image.open(img_global)
        img_global = img_global.convert('RGB')
        global_tensor = self.trans(img_global)

        img_back = Image.open(img_back)
        img_back = img_back.convert('RGB')
        back_tensor = self.trans(img_back)

        face_rtv = Image.open(face_rtv)
        face_rtv = face_rtv.convert('RGB')
        face_rtv_tensor = self.trans(face_rtv)

        gt_bvp = Image.open(gt_bvp)
        gt_bvp = gt_bvp.resize((1, temporal_size))
        gt_bvp = np.array(gt_bvp)

        neg_for_back = get_random_bvp(self.gt_bvps)

        return face_tensor, global_tensor, back_tensor, face_rtv_tensor, gt_bvp, neg_for_back

    def __len__(self):
        return len(self.img_faces)


def loader_data():
    img_face_names = []
    img_global_names = []
    img_back_names = []

    face_rtv_names = []

    gt_map_names = []

    if args.dataset_name == 'the_nirp':
        the_nirp = ['06_nirp_indoor', '07_nirp_car940', '08_nirp_car975']
        for i in range(len(the_nirp)):
            root_dir = '../traffic_dataset/' + the_nirp[i] + '/'

            if input_img_pattern == 'yuv':
                img_face_dir = root_dir + 'yuv_face/'
                face_rtv_dir = root_dir + 'yuv_face/'
                if args.whether_rtv is True:
                    img_global_dir = root_dir + 'yuv_glob_rtv/'
                    img_back_dir = root_dir + 'yuv_without_face_rtv/'
                else:
                    img_global_dir = root_dir + 'yuv_glob/'
                    img_back_dir = root_dir + 'yuv_without_face/'
            elif input_img_pattern == 'rgb':
                img_face_dir = root_dir + 'norm_face/'
                face_rtv_dir = root_dir + 'norm_face_rtv/'
                if args.whether_rtv is True:
                    img_global_dir = root_dir + 'norm_glob_rtv/'
                    img_back_dir = root_dir + 'norm_without_face_rtv/'
                else:
                    img_global_dir = root_dir + 'norm_glob/'
                    img_back_dir = root_dir + 'norm_without_face/'

            img_face_list = os.listdir(img_face_dir)
            img_face_list.sort()
            for img_0 in img_face_list:
                img_path = img_face_dir + img_0
                if img_path.endswith('.png'):
                    img_face_names.append(img_path)

            img_global_list = os.listdir(img_global_dir)
            img_global_list.sort()
            for img_0 in img_global_list:
                img_path = img_global_dir + img_0
                if img_path.endswith('.png'):
                    img_global_names.append(img_path)

            img_back_list = os.listdir(img_back_dir)
            img_back_list.sort()
            for img_0 in img_back_list:
                img_path = img_back_dir + img_0
                if img_path.endswith('.png'):
                    img_back_names.append(img_path)

            face_rtv_list = os.listdir(face_rtv_dir)
            face_rtv_list.sort()
            for img_0 in face_rtv_list:
                img_path = face_rtv_dir + img_0
                if img_path.endswith('.png'):
                    face_rtv_names.append(img_path)

            gt_map_dir = root_dir + 'gt_map/'
            gt_map_list = os.listdir(gt_map_dir)
            gt_map_list.sort()
            for img_0 in gt_map_list:
                img_path = gt_map_dir + img_0
                if img_path.endswith('.png'):
                    gt_map_names.append(img_path)

    else:
        root_dir = '../traffic_dataset/' + args.dataset_name + '/'

        if input_img_pattern == 'yuv':
            img_face_dir = root_dir + 'yuv_face/'
            face_rtv_dir = root_dir + 'yuv_face/'
            if args.whether_rtv is True:
                img_global_dir = root_dir + 'yuv_glob_rtv/'
                img_back_dir = root_dir + 'yuv_without_face_rtv/'
            else:
                img_global_dir = root_dir + 'yuv_glob/'
                img_back_dir = root_dir + 'yuv_without_face/'
        elif input_img_pattern == 'rgb':
            img_face_dir = root_dir + 'norm_face/'
            face_rtv_dir = root_dir + 'norm_face_rtv/'
            if args.whether_rtv is True:
                img_global_dir = root_dir + 'norm_glob_rtv/'
                img_back_dir = root_dir + 'norm_without_face_rtv/'
            else:
                img_global_dir = root_dir + 'norm_glob/'
                img_back_dir = root_dir + 'norm_without_face/'

        img_face_list = os.listdir(img_face_dir)
        img_face_list.sort()
        for img_0 in img_face_list:
            img_path = img_face_dir + img_0
            if img_path.endswith('.png'):
                img_face_names.append(img_path)

        img_global_list = os.listdir(img_global_dir)
        img_global_list.sort()
        for img_0 in img_global_list:
            img_path = img_global_dir + img_0
            if img_path.endswith('.png'):
                img_global_names.append(img_path)

        img_back_list = os.listdir(img_back_dir)
        img_back_list.sort()
        for img_0 in img_back_list:
            img_path = img_back_dir + img_0
            if img_path.endswith('.png'):
                img_back_names.append(img_path)

        face_rtv_list = os.listdir(face_rtv_dir)
        face_rtv_list.sort()
        for img_0 in face_rtv_list:
            img_path = face_rtv_dir + img_0
            if img_path.endswith('.png'):
                face_rtv_names.append(img_path)

        gt_map_dir = root_dir + 'gt_map/'
        gt_map_list = os.listdir(gt_map_dir)
        gt_map_list.sort()
        for img_0 in gt_map_list:
            img_path = gt_map_dir + img_0
            if img_path.endswith('.png'):
                gt_map_names.append(img_path)

    assert len(img_face_names) == len(img_global_names)
    assert len(img_face_names) == len(img_back_names)
    assert len(img_face_names) == len(face_rtv_names)
    assert len(img_face_names) == len(gt_map_names)

    if args.limit_train_scale is True:
        train_data_scale = int(args.train_data_scale)
        img_face_names = img_face_names[0: train_data_scale]
        img_global_names = img_global_names[0: train_data_scale]
        img_back_names = img_back_names[0: train_data_scale]
        face_rtv_names = face_rtv_names[0: train_data_scale]
        gt_map_names = gt_map_names[0: train_data_scale]

    transform = DataAugmentation()

    trans_ds = GenDataset(img_faces=img_face_names,
                          img_globals=img_global_names,
                          img_backs=img_back_names,
                          face_rtvs=face_rtv_names,
                          gt_bvps=gt_map_names,
                          trans=transform)

    train_learning_set = data.DataLoader(dataset=trans_ds,
                                         batch_size=batch_size,
                                         shuffle=args.data_shuffle,
                                         drop_last=True)

    return train_learning_set


train_dl = loader_data()


if args.network_name == 'DeepPhys':
    model = DeepPhys(frames=temporal_size)
elif args.network_name == 'DualGAN':
    model = DualGAN(frames=temporal_size)
elif args.network_name == 'TS_CAN':
    model = TS_CAN(frames=temporal_size)
elif args.network_name == 'PFE_TFA':
    model = PFE_TFA(frames=temporal_size, device_ids=[0])
elif args.network_name == 'NEST':
    model = NEST()
elif args.network_name == 'ND_DeepRPPG':
    model = ND_DeepRPPG(frames=temporal_size)
elif args.network_name == 'rPPG_HiBa':
    model = rPPG_HiBa()
elif args.network_name == 'DD_rPPGNet':
    model = DD_rPPGNet(conv_type='LDC_M')
    freq_contrast = FrequencyContrast(model, args.device)
    loss_func_rPPG = ContrastLoss(delta_t=temporal_size // 2, K=4, Fs=30,
                                  high_pass=40, low_pass=250,
                                  PSD_or_signal='PSD')
    loss_func_rPPG_Sig = ContrastLoss(delta_t=temporal_size // 2, K=4, Fs=30,
                                      high_pass=40, low_pass=250,
                                      PSD_or_signal='signal')

elif args.network_name == 'EfficientPhys':
    model = EfficientPhys_Transformer()
elif args.network_name == 'PhysFormer':
    model = PhysFormer()
elif args.network_name == 'RhythmFormer':
    model = RhythmFormer()
elif args.network_name == 'NDNet':
    model = NDNet()

elif args.network_name == 'RhythmMamba':
    model = RhythmMamba()
elif args.network_name == 'PhysMambaTD':
    model = PhysMambaTD(frames=temporal_size)
elif args.network_name == 'PhysMambaCASSD':
    model = PhysMambaCASSD()

elif args.network_name == 'Our_Net':
    model = Our_NDNet(temporal_size=temporal_size, img_size=img_size, positive_num=4,
                      in_chans=3, embed_dim=96, out_chans=1, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=8)

model_weight = './training_save_dir/' + args.network_name + '/check_' + args.dataset_name + '_' + str(args.epochs).zfill(3) + '_' + args.save_weight_name


if torch.cuda.device_count() > 1:
    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids)
    if args.pretraining is True:
        model.load_state_dict(torch.load(model_weight))
    model.to(args.device)
elif torch.cuda.device_count() == 1:
    if args.pretraining is True:
        model.load_state_dict(torch.load(model_weight))
    model = model.cuda()

optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), eps=1e-8)

criterion_wave = NegPearson()
criterion_gan = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_cl = FCL(Fs=video_fps)

if args.save_train_log is True:
    train_log = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_log = train_log.add_sheet('sheet', cell_overwrite_ok=True)


for epoch in range(args.epochs):
    train_tqdm = tqdm.tqdm(iterable=train_dl, total=len(train_dl))
    train_tqdm.set_description('train epoch {:2d}'.format(epoch))

    for step, data in enumerate(train_tqdm):
        img_faces, img_globs, img_backs, face_rtvs, gt_bvps, neg_for_backs = data

        img_face = img_to_device(img_faces, c=3)
        img_glob = img_to_device(img_globs, c=3)
        img_back = img_to_device(img_backs, c=3)
        face_rtv = img_to_device(face_rtvs, c=3)

        gt_bvp = gt_bvps.to(args.device)
        gt_bvp = gt_bvp.view(batch_size, temporal_size)
        gt_bvp = gt_bvp.float()
        gt_bvp = (gt_bvp - torch.mean(gt_bvp)) / torch.std(gt_bvp)

        neg_for_back = neg_for_backs.to(args.device)
        neg_for_back = neg_for_back.view(batch_size, 4, temporal_size)
        neg_for_back = neg_for_back.float()
        neg_for_back = (neg_for_back - torch.mean(neg_for_back)) / torch.std(neg_for_back)

        if args.network_name == 'DeepPhys':
            estimated_bvp = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

        elif args.network_name == 'TS_CAN':
            estimated_bvp = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

        elif args.network_name == 'DualGAN':
            noise_bvp = torch.randn(batch_size, temporal_size)
            noise_bvp = noise_bvp.to(args.device)
            noise_bvp = noise_bvp.float()
            real_label = torch.ones(batch_size, 1).to(args.device)
            fake_label = torch.zeros(batch_size, 1).to(args.device)
            estimated_bvp, reconstruct_wave, fake_1, fake_2, real_1 = model(img_face, gt_bvp, noise_bvp)
            loss_bvp = criterion_wave(estimated_bvp, gt_bvp)
            loss_wave_reconstruct = criterion_wave(reconstruct_wave, gt_bvp)
            d_real_loss = criterion_gan(real_1, real_label)
            d_fake_loss_1 = criterion_gan(fake_1, fake_label)
            d_fake_loss_2 = criterion_gan(fake_2, fake_label)
            loss = loss_bvp * 0.2 + loss_wave_reconstruct * 0.2 + d_real_loss * 0.2 + d_fake_loss_1 * 0.2 + d_fake_loss_2 * 0.2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_bvp_value = loss_bvp.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f} loss_wave={loss_bvp_value:.5f}'

        elif args.network_name == 'PFE_TFA':
            estimated_bvp_1, estimated_bvp_2 = model(img_face)
            loss_1 = criterion_wave(estimated_bvp_1, gt_bvp)
            loss_2 = criterion_wave(estimated_bvp_2, gt_bvp)
            loss = loss_1 * 0.25 + loss_2 * 0.25
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_wave_1_value = loss_1.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f} loss_wave={loss_wave_1_value:.5f}'

        elif args.network_name == 'NEST':
            estimated_bvp, _, _ = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

        elif args.network_name == 'rPPG_HiBa':
            estimated_bvp, _, _ = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

        elif args.network_name == 'ND_DeepRPPG':
            real_label = torch.ones(batch_size, 1).to(args.device)
            fake_label = torch.zeros(batch_size, 1).to(args.device)
            estimated_bvp, noise_bvp, background_bvp, x_real, x_fake = model(img_face, img_back)
            loss_bvp = criterion_wave(estimated_bvp, gt_bvp)
            loss_noise = criterion_l1(noise_bvp, background_bvp)
            d_real_loss = criterion_gan(x_real, real_label)
            d_fake_loss = criterion_gan(x_fake, fake_label)
            loss = loss_bvp * 0.25 + loss_noise * 0.25 + d_real_loss * 0.25 + d_fake_loss * 0.25
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_bvp_value = loss_bvp.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs} loss={loss_value:.5f} loss_wave={loss_bvp_value:.5f}'

        elif args.network_name == 'DD_rPPGNet':
            b, c, t, s = img_face.shape
            img_face = F.interpolate(img_face, size=(t, s ** 2), mode='bilinear', align_corners=False)
            img_face = img_face.view(b, c, t, s, s)

            img_back = F.interpolate(img_back, size=(t, s ** 2), mode='bilinear', align_corners=False)
            img_back = img_back.view(b, c, t, s, s)

            multi_spatial_y, multi_spatial_noise_y, multi_spatial_noisy_rPPG_y, fg_to_bg_noise = model(img_face, img_back)

            multi_spatial_y_aug = model(img_face, None, if_de_interfered=True)
            multi_spatial_noisy_y_aug = model(img_face, None, if_de_interfered=False)

            loss_noise_np = criterion_wave(multi_spatial_noise_y[:, -1], fg_to_bg_noise[:, -1])

            all_noises = torch.cat([multi_spatial_noise_y, fg_to_bg_noise], dim=0)
            all_noises = rearrange(all_noises, 'b s d -> (b s) d')
            clusters = {}
            n_clusters = 4
            noise_labels, noise_centroids = KMeans(all_noises, n_clusters=n_clusters)
            for i in range(n_clusters):
                clusters[i] = all_noises[noise_labels == i]

            loss_kmeans = 0
            for i in range(n_clusters):
                anc = clusters[i]
                if anc.shape[0] < 2:
                    continue

                neg = torch.cat([clusters[j] for j in range(n_clusters) if j != i], dim=0)

                anc = anc.unsqueeze(1)
                neg = neg.unsqueeze(1)
                print(f"{anc.shape=}, {neg.shape=}")

                if anc.shape[0] < 2 or neg.shape[0] < 2:
                    continue

                loss_tmp = loss_func_rPPG_Sig.forward_k_means(anc, neg)
                loss_kmeans += loss_tmp

            all_noisy_ppg = torch.cat([multi_spatial_noisy_rPPG_y, multi_spatial_noisy_y_aug], dim=1)
            loss_contrastive_cr = loss_func_rPPG(all_noisy_ppg)

            all_ppg = torch.cat([multi_spatial_y, multi_spatial_y_aug], dim=1)
            loss_contrastive_dcr = loss_func_rPPG.forward_pos_and_neg(all_ppg, fg_to_bg_noise)

            y_a, y_p, y_n = freq_contrast(img_face, img_back)
            loss_freq_contrast = loss_func_rPPG.forward_anc_pos_neg(y_a, y_p, y_n)

            total_loss = loss_noise_np * 0.05 + loss_kmeans * 0.05 + loss_contrastive_cr * 0.5 + loss_contrastive_dcr + loss_freq_contrast * 0.5
            loss = loss_noise_np

            loss_view = loss.clone()
            nonzero_finite_vals = torch.masked_select(
                loss, torch.isfinite(loss) & loss.ne(0)
            )

            optimizer.zero_grad()
            nonzero_finite_vals.backward()
            optimizer.step()
            print(loss)
            nonzero_finite_value = nonzero_finite_vals.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={nonzero_finite_value:.5f}'

            """ Transformer and Mamba """

        elif args.network_name == 'EfficientPhys':
            estimated_bvp = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

        elif args.network_name == 'PhysFormer':
            estimated_bvp, _, _, _ = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

        elif args.network_name == 'RhythmFormer':
            estimated_bvp = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

        elif args.network_name == 'NDNet':
            gen_map, estimated_bvp_1, estimated_bvp_2 = model(img_face, img_back, img_glob)
            gt_map = gt_bvp.view(batch_size, 1, temporal_size, 1)
            gt_map = gt_map.expand(-1, -1, -1, img_size)
            loss_map = criterion_l1(gen_map, gt_map)
            loss_wave_1 = criterion_wave(estimated_bvp_1, gt_bvp)
            loss_wave_2 = criterion_wave(estimated_bvp_2, gt_bvp)
            loss_total = loss_wave_1 * 0.5 + loss_wave_2 * 0.25 + loss_map * 0.25
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            loss_value = loss_total.item()
            loss_wave_value = loss_wave_1.item()
            train_tqdm.desc = f'epoch [{epoch + 1}//{args.epochs}] loss={loss_total:.5f} loss_wave={loss_wave_value:.5f}'

        elif args.network_name == 'RhythmMamba':
            estimated_bvp = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

        elif args.network_name == 'PhysMambaCASSD':
            estimated_bvp = model(img_face)
            loss = criterion_wave(estimated_bvp, gt_bvp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_tqdm.desc = f'[{epoch + 1}//{args.epochs}] loss={loss_value:.5f}'

            """ Our model """

        elif args.network_name == 'Our_Net':
            estimated_bvp, back_tensor, rtv_tensor = model(img_face, img_back, face_rtv)

            loss_wave = criterion_wave(estimated_bvp, gt_bvp)
            a1 = randint(0, 3)
            a2 = randint(0, 3)

            back_tensor_1 = back_tensor[:, a1, :]
            back_tensor_2 = back_tensor[:, a2, :]

            neg_for_back = neg_for_back.view(4, batch_size, temporal_size)
            neg_for_face = rtv_tensor.view(4, batch_size, temporal_size)

            loss_cl_back = criterion_cl(neg_for_back, back_tensor_1, back_tensor_2)
            loss_cl_face = criterion_cl(neg_for_face, estimated_bvp, gt_bvp)

            loss_total = loss_wave + loss_cl_back + loss_cl_face

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            loss_value = loss_total.item()
            loss_wave_value = loss_wave.item()
            cl_back_value = loss_cl_back.item()
            cl_face_value = loss_cl_face.item()
            train_tqdm.desc = f'epoch [{epoch + 1}//{args.epochs}] l={loss_value:.5f} l_bvp={loss_wave_value:.5f}' \
                              f' l_back={cl_back_value:.5f} l_face={cl_face_value:.5f}'

        if args.save_train_log is True:
            sheet_log.write(epoch, 0, loss_value)

    if args.save_weight is True:
        torch.save(model.state_dict(),
                   './training_save_dir/' + args.network_name + '/check_' + args.dataset_name + '_' + str(args.epochs).zfill(3) + '_' + args.save_weight_name)

    if args.save_train_log is True:
        train_log.save('./training_save_dir/' + args.network_name + '/train_log_' + args.dataset_name + '.xls')