# -*- coding=utf-8 -*-
import os
import random
import nibabel as nib
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tasks.aneurysm.datasets.data_utils import *
from tasks.aneurysm.datasets.aneurysm_dataset import ANEURYSM_SEG
from utils.config import cfg
from utils.tools.util import progress_monitor
from .coord_conv_np import *


def read_data(subject):
    ''' input: pid of data, output: aneurysm image & lesion mask and vessel
    More Info: dataset organization as follow
    eg: subject-0649385
    - image: 0649385.nii.gz
    - aneurysm lesion: 0649385_aneurysm.nii.gz / 0649385_mask.nii.gz
    '''
    img_path = os.path.join(cfg.TRAIN.DATA.NII_FOLDER, '%s.nii.gz' % subject)
    nii_img = nib.load(img_path)
    img = nii_img.get_data()
    spacing_x, spacing_y, spacing_z = nii_img.header['pixdim'][1:4]  # nii data extract xyz spacing info
    spacing_zxy = spacing_z, spacing_x, spacing_y,

    aneurysm_path = os.path.join(cfg.TRAIN.DATA.ANEURYSM_FOLDER, '%s_mask.nii.gz' % subject)
    if os.path.exists(aneurysm_path):
        aneurysm = nib.load(aneurysm_path).get_data()
    else:
        aneurysm = np.zeros(img.shape, 'uint8')


    # transpose dimensions of input array: xyz => zxy
    # for z axes, slice index from 0 in neck increase to head-top
    # for slice image, set (x,y) axes order let man face toward left, not front
    img = np.transpose(img, (2, 0, 1))
    aneurysm = np.transpose(aneurysm, (2, 0, 1))

    return img, aneurysm, spacing_zxy


def sampe_subjects(subjects, aneurysm_bbox, patch_num):
    img_lst, gt_lst, delayed_transform_add_coords = [], [], []
    for subject in subjects:
        image, aneurysm, spacing_zxy = read_data(subject)  # origin data
        img, gt, _delayed_transform_add_coords = AneurysmSampler(subject,  aneurysm_bbox[subject], spacing_zxy, image, aneurysm).sample(patch_num)
        img_lst.append(img)
        gt_lst.append(gt)
        delayed_transform_add_coords += _delayed_transform_add_coords

    return np.concatenate(img_lst), np.concatenate(gt_lst), delayed_transform_add_coords


def get_aneurysm_bbox():
    '''append invalid aneurysm bbox coords filter'''
    aneurysm_bbox = {}
    with open(cfg.TRAIN.DATA.ANEURYSM_BBOX, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines, start=1):
            vv = line.strip().split(' ')
            '''check aneurysm bbox coords'''
            coord = [int(v) for v in vv[1:]]
            if len(coord) != 6:
                print('except data in: %d %s' % (index, line))
                continue
            z, x, y, d, w, h = coord
            if any(_ < 0 for _ in (z, x, y)) or any(_ < 1 for _ in (d, w, h)):
                print('valid bbox coords info:', line)
                continue

            if vv[0] not in aneurysm_bbox:
                aneurysm_bbox[vv[0]] = []
            aneurysm_bbox[vv[0]].append(coord)
    return aneurysm_bbox


class AneurysmDataset(Thread):

    def __init__(self, train_lst):
        super().__init__()

        self.train_lst = train_lst
        with open(self.train_lst, 'r') as f:
            lines = f.readlines()
            self.subjects = [line.strip() for line in lines]

        self.aneurysm_bbox = get_aneurysm_bbox()
        for subject in self.subjects:
            if subject not in self.aneurysm_bbox:
                self.aneurysm_bbox[subject] = []

        self.ex = None
        self.objs = []

    def asyn_sample(self, subject_num=100, patch_num=100, max_workers=20):
        indices = list(range(len(self.subjects)))
        random.shuffle(indices)
        indices = indices[:subject_num]

        monitor = progress_monitor(total=len(indices))
        subjects = [self.subjects[idx] for idx in indices]
        self.ex = ProcessPoolExecutor(max_workers=max_workers)
        self.objs = []

        step = 1
        for i in range(0, len(subjects), step):
            future = self.ex.submit(sampe_subjects, subjects[i:i + step], self.aneurysm_bbox, patch_num)
            future.add_done_callback(fn=monitor)
            self.objs.append(future)
        print('data processing in async ...')

    def get_data_loader(self):
        '''wait for async processing finished, then extract return aneurysm image and mask data'''
        self.ex.shutdown(wait=True)
        img_lst, gt_lst, label_lst = [], [], []
        delayed_transform_add_coords = []
        for obj in self.objs:
            img, gt, _delayed_transform_add_coords = obj.result()
            img_lst.append(img)
            gt_lst.append(gt)
            label_lst.append([int((_ > 0).sum() >= 10) for _ in gt])
            delayed_transform_add_coords += _delayed_transform_add_coords

        t0 = time.time()
        # img: b,c,d,w,h
        # gt: b,d,w,h
        img, gt, label = np.concatenate(img_lst), np.concatenate(gt_lst), np.concatenate(label_lst)

        print('concate need %.4f' % (time.time() - t0))
        para_dict = {"image": img, "gt": gt, 'label:': label, \
                     'delayed_transform_add_coords': delayed_transform_add_coords}
        return ANEURYSM_SEG(para_dict, "train")


class AneurysmSampler(object):
    __slots__ = 'subject','image','aneurysm','img_size','patch_size','offset',\
                'bbox_list','spacing_zxy','is_neck','add_coords','slice_threshold','slice_zs','slice_ze'
    def __init__(self, subject, bbox_list, spacing_zxy, image, aneurysm, is_neck=True, add_coords=False):
        self.subject = subject
        self.image = image
        self.aneurysm = aneurysm
        self.img_size = aneurysm.shape
        self.patch_size = cfg.TRAIN.DATA.PATCH_SIZE
        self.offset = cfg.TRAIN.DATA.PATCH_OFFSET
        self.bbox_list = bbox_list
        self.spacing_zxy = spacing_zxy
        self.is_neck = is_neck
        # self.add_coords = add_coords
        # self.add_coords = False if 'ADD_COORDS' not in cfg.MODEL else cfg.MODEL.ADD_COORDS
        self.add_coords = False
        # z axes all slice: self.img_size[0]; train/eval used the layers of head-top in threshold,
        # here we use 18cm for default to ensure these layers included
        self.slice_threshold = max(0, self.img_size[0] - int(180 / self.spacing_zxy[0]))
        self.slice_zs = max(0, self.img_size[0] - int(180 / self.spacing_zxy[0]))  # [5,15] [6, 14]
        self.slice_ze = max(0, self.img_size[0] - int(20 / self.spacing_zxy[0]))

        self.image = self.image[np.newaxis, np.newaxis]

    def sample_positive_coords(self, bbox, k):
        x_0, y_0, z_0, w, h, d = bbox
        x_1, y_1, z_1 = x_0 + w, y_0 + h, z_0 + d

        p_size = (
            max(self.patch_size[0], w + 2 * self.offset[0]),
            max(self.patch_size[1], h + 2 * self.offset[1]),
            max(self.patch_size[2], d + 2 * self.offset[2]),
        )

        x0 = max(0, x_1 + self.offset[0] - p_size[0])
        y0 = max(0, y_1 + self.offset[1] - p_size[1])
        z0 = max(0, z_1 + self.offset[2] - p_size[2])
        x1 = max(0, min(x_0 - self.offset[0], self.img_size[0] - p_size[0]))
        y1 = max(0, min(y_0 - self.offset[1], self.img_size[1] - p_size[1]))
        z1 = max(0, min(z_0 - self.offset[2], self.img_size[2] - p_size[2]))

        if x0 > x1 or y0 > y1 or z0 > z1:
            print(x_0, x_1, y_0, y_1, z_0, z_1)
            print(self.subject, 'error')
            return [], []

        coords_list = []
        for i in range(k):
            x = random.randint(x0, x1)
            y = random.randint(y0, y1)
            z = random.randint(z0, z1)

            b1 = x, y, z, p_size[0], p_size[1], p_size[2]
            is_b1_contain_b2 = False
            for b2 in self.bbox_list:
                if bbox3d_contain(b1, b2):  # b1 包含 b2
                    is_b1_contain_b2 = True
            assert is_b1_contain_b2, \
                'sample positive bbox error:{}, {},{}, {}'.format(self.subject, bbox, b1, self.bbox_list,)

            x_offset = random.randint(0, p_size[0] - self.patch_size[0])
            y_offset = random.randint(0, p_size[1] - self.patch_size[1])
            z_offset = random.randint(0, p_size[2] - self.patch_size[2])

            _x0, _y0, _z0 = x + x_offset, y + y_offset, z + z_offset
            coords_list += [(_x0, _y0, _z0)]
        return coords_list

    def sample_negative_random(self, k):

        max_try_times = 10000
        neg_coords = []
        neg_num = 0
        for try_iter in range(max_try_times):

            x = random.randint(0, self.img_size[0] - self.patch_size[0])
            y = random.randint(0, self.img_size[1] - self.patch_size[1])
            z = random.randint(0, self.img_size[2] - self.patch_size[2])

            b1 = [x, y, z, self.patch_size[0],
                  self.patch_size[1], self.patch_size[2]]
            r = 0
            for b2 in self.bbox_list:
                r = max(bbox3d_ratio(b1, b2), r)

            if r < 0.5:
                if self.is_neck:
                    coord = [(x, y, z)] if self.slice_zs <= (x + self.patch_size[0] // 2) <= self.slice_ze else []
                    if coord:
                        neg_coords.append(coord[0])
                        neg_num += 1
                else:
                    neg_coords.append((x, y, z))
                    neg_num += 1

            if neg_num >= k:
                break

        return neg_coords

    def sample_negative_traverse(self, k):
        # same to the process of patch coords generated, and stride=patch size // 2
        coords = get_patch_coords(self.patch_size, self.img_size, 2)
        if self.is_neck:
            coords = [_ for _ in coords if self.slice_zs <= (_[0] + self.patch_size[0] // 2) <= self.slice_ze]
        random.shuffle(coords)
        neg_coords = []
        neg_num = 0
        for idx, coord in enumerate(coords):
            b1 = [coord[0], coord[1], coord[2],
                  self.patch_size[0], self.patch_size[1], self.patch_size[2]]
            r = 0
            for b2 in self.bbox_list:
                r = max(bbox3d_ratio(b1, b2), r)

            if r < 0.2:
                neg_coords.append(coord)
                neg_num += 1

            if neg_num >= k:
                break

        return neg_coords

    def sample(self, k):
        img_lst, gt_lst, delayed_transform_add_coords = [], [], []
        positive_coords_list = []
        for bbox in self.bbox_list:
            _coords_list = self.sample_positive_coords(bbox, k)
            positive_coords_list += _coords_list

        if self.add_coords:
            addcoords = AddCoordsNp4GCP(rank=3, size=self.img_size, with_r=False)
        wl, ww, = cfg.TRAIN.DATA.WL_WW

        for x0,y0,z0 in positive_coords_list:
            x1, y1, z1 = x0 + self.patch_size[0], y0 + self.patch_size[1], z0 + self.patch_size[2]
            img = self.image[..., x0:x1, y0:y1, z0:z1]
            img = set_window_wl_ww(img, ww=ww, wl=wl)  # 1,1,Z,XY
            img = (img / 255.0) * 2.0 - 1.0
            if self.add_coords:
                delayed_transform_add_coords += [(addcoords, (x0, y0, z0))]
            ane = self.aneurysm[..., x0:x1, y0:y1, z0:z1]
            gt = (ane > 0).astype(np.uint8)
            img_lst += [img]
            gt_lst += [gt[np.newaxis]]

        neg_num = max(k, len(img_lst))
        negative_coords_list = []
        negative_coords_list.extend(self.sample_negative_traverse(neg_num // 2))  # Priority 1
        # Priority 2: full random coords for negative patch
        negative_coords_list.extend(self.sample_negative_random(neg_num - neg_num // 2))

        for x0,y0,z0 in negative_coords_list:
            x1, y1, z1 = x0 + self.patch_size[0], y0 + self.patch_size[1], z0 + self.patch_size[2]
            img = self.image[..., x0:x1, y0:y1, z0:z1]
            img = set_window_wl_ww(img, ww=ww, wl=wl)  # 1,1,Z,XY
            img = (img / 255.0) * 2.0 - 1.0
            if self.add_coords:
                delayed_transform_add_coords += [(addcoords, (x0, y0, z0))]
            gt = np.zeros(img.shape[-3:], dtype=np.uint8)
            img_lst += [img]
            gt_lst += [gt[np.newaxis]]

        img_lst, gt_lst = np.concatenate(img_lst), np.concatenate(gt_lst)

        return img_lst, gt_lst, delayed_transform_add_coords