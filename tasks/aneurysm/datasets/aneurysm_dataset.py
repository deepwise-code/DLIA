from utils.config import cfg
from tasks.aneurysm.datasets.base_dataset import BaseDataset
from tasks.aneurysm.datasets.data_utils import *
from tasks.aneurysm.coord_conv_np import *

import numpy as np
import torch
import nibabel as nib
import os


class ANEURYSM_SEG(BaseDataset):
    def __init__(self, para_dict, stage):
        super(ANEURYSM_SEG, self).__init__(para_dict, stage)

    def train_init(self):
        self.image = self.para_dict.get("image", None)
        self.gt = self.para_dict.get("gt", None)
        self.transform_add_coords = self.para_dict.get("delayed_transform_add_coords", None)
        self.flip = False
        self.num = len(self.image)
        self.use_multi_channel = self.image.shape[1] == 4  # b,c,d,w,h, c!=1
        self.add_coords = False if 'ADD_COORDS' not in cfg.MODEL else cfg.MODEL.ADD_COORDS

    def test_init(self):
        self.subject = self.para_dict.get("subject", None)
        nii_image = nib.load(os.path.join(cfg.TEST.DATA.NII_FOLDER,
                                          "{}.nii.gz".format(self.subject)))
        img = nii_image.get_data()
        WL, WW = cfg.TRAIN.DATA.WL_WW
        img = set_window_wl_ww(img, WL, WW)
        img = (img / 255.0) * 2.0 - 1.0
        self.img = np.transpose(img, (2, 0, 1))  # (x, y, z) => (z,x,y) order
        self.img = self.img[np.newaxis, np.newaxis]  # 1,1,Z,XY
        add_coords = False if 'ADD_COORDS' not in cfg.MODEL else cfg.MODEL.ADD_COORDS
        # print('add coords:', add_coords)
        if add_coords:
            self.img = AddCoordsNp(rank=3, with_r=False)(self.img)  # 1,1,Z,XY => 1,4,Z,XY

        self.patch_size = cfg.TEST.DATA.PATCH_SIZE
        # self.coords = get_patch_coords(self.patch_size, self.img.shape)

        # extract voxel sample spacing info to calculate slice range for evaluation
        x_spacing, y_spacing, z_spacing = nii_image.header['pixdim'][1:4]
        self.zxy_spacing = (z_spacing, x_spacing, y_spacing,)
        self.affine = nii_image.affine
        d, w, h = self.img.shape[-3:]
        z_axes_height = 180  # EXP: 18CM
        z_axes_ignore = 20  #
        if 0.1 < z_spacing <= 1.:
            z_axes_start = max(0, d - int(z_axes_height / z_spacing))
            z_axes_end = max(0, d - int(z_axes_ignore / z_spacing))
        else:
            self.logger.info('invalid z_spacing not in [0.1, 1.]: {} {:.2f}'.format(self.subject, z_spacing))
            z_axes_start = 0
            z_axes_end = d - 1
        volum_zxy = self.img.shape[-3:]
        patch_zxy = cfg.TEST.DATA.PATCH_SIZE
        patch_step_zxy = [_//2 for _ in patch_zxy]
        specify_z_axes_range = (z_axes_start, z_axes_end)
        self.coords = gen_patch_coords_ext(volum_zxy, patch_zxy, patch_step_zxy, specify_z_axes_range)
        self.num = len(self.coords)

    def train_load(self, index):
        return self.train_load_multi_channel(index)

    def train_load_single_channel(self, index):
        image, gt = self.image[index], self.gt[index]
        image = (image / 255.0) * 2.0 - 1.0
        label = 1 if gt.sum() > 0 else 0

        if self.flip:
            flip_x = np.random.choice(2) * 2 - 1
            flip_y = np.random.choice(2) * 2 - 1
            flip_z = np.random.choice(2) * 2 - 1
            image = image[::flip_x, ::flip_y, ::flip_z]
            gt = gt[::flip_x, ::flip_y, ::flip_z]

        image = image.astype('float32')
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        gt = torch.from_numpy(gt)
        data = {'img': image, 'gt': gt, 'label': label}
        return data

    def train_load_multi_channel(self, index):
        image, gt = self.image[index], self.gt[index]
        # image = (image / 255.0) * 2.0 - 1.0
        if self.add_coords:
            patch_addcoords_transform, params = self.transform_add_coords[index]
            image = patch_addcoords_transform(image[np.newaxis, ...], params)[0]


        label = 1 if gt.sum() > 0 else 0

        image = image.astype('float32')
        image = torch.from_numpy(image)
        gt = torch.from_numpy(gt)
        data = {'img': image, 'gt': gt, 'label': label}
        return data

    def test_load(self, index):
        x, y, z = self.coords[index]
        img = self.img[
              ...,
              x:x + self.patch_size[0],
              y:y + self.patch_size[1],
              z:z + self.patch_size[2],
        ]
        img = img.astype('float32')
        # img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img[0])
        coord = np.array([[x, x + self.patch_size[0]],
                          [y, y + self.patch_size[1]],
                          [z, z + self.patch_size[2]]])
        coord = torch.from_numpy(coord)
        return img, coord

    def volume_size(self):
        return self.img.shape[-3:]

    def get_other_infos(self):
        info_map = {
            'zxy_shape': self.img.shape[-3:],
            'zxy_spacing': self.zxy_spacing,
            'affine': self.affine,  # for save result in nii formation
        }
        return info_map

    def save(self, seg, save_dir):
        '''save seg result to specify directory
        :param seg: segmentation result, type ndarray
        :param save_dir: nii file save directory
        :return: None
        '''
        save_path = os.path.join(save_dir, '%s_seg.nii.gz' % self.subject)
        seg = np.transpose(seg, (1, 2, 0))  # (z, x, y) => (x, y, z)
        affine = self.affine
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, save_path)
