# -*- coding: utf-8 -*-
"""
   Author :       lirenqiang
   date：          2019/9/25
"""
import numpy as np


def set_window_wl_ww(tensor, wl=225, ww=450):
    w_min, w_max = wl - ww // 2, wl + ww // 2
    tensor[tensor < w_min] = w_min
    tensor[tensor > w_max] = w_max
    tensor = ((1.0 * (tensor - w_min) / (w_max - w_min)) * 255).astype(np.uint8)

    return tensor


def filter_negative_coords(coords: list, layer_total: int, layer_nums=300, layer_percent=0.5):
    """
    :param coords:
    :param layer_total:
    :param layer_nums:
    :param layer_percent:
    :return: filtered coords
    """
    save_layers = max(layer_total * layer_percent, layer_nums)
    slice_start = max(0, layer_total - save_layers)
    coords_ = [_ for _ in coords if _[0] >= slice_start]
    # print('before: %d, after: %d' %(len(coords), len(coords_)))
    return coords_


def get_patch_coords(patch_xyz, volume_xyz, overlap=0, stride=(2, 2, 2)):
    """[0:IMG_SIZE-PATCH_SIZE: stride]  方式遍历生成坐标"""
    coords = []
    p_x, p_y, p_z = patch_xyz[0], patch_xyz[1], patch_xyz[2]
    v_x, v_y, v_z = volume_xyz[0], volume_xyz[1], volume_xyz[2]
    x = 0

    if overlap > 0:
        sx, sy, sz = p_x // overlap, p_y // overlap, p_z // overlap
    else:
        sx, sy, sz = stride

    while x < v_x:
        y = 0
        while y < v_y:
            z = 0
            while z < v_z:
                coords.append(
                    (x if x + p_x < v_x else v_x - p_x,
                     y if y + p_y < v_y else v_y - p_y,
                     z if z + p_z < v_z else v_z - p_z)
                )
                if z + p_z >= v_z:
                    break
                z += sz
            if y + p_y >= v_y:
                break
            y += sy
        if x + p_x >= v_x:
            break
        x += sx

    return coords


def gen_patch_coords_ext(volum_zxy, patch_zxy, patch_step_zxy, specify_z_axes_range=None):
    '''z_axes: assume the start slice index 0 at neck, and increase to head-top(end)
    return patch coords list with item of rectangle left-top point  z_xy order'''
    VZ, VX, VY = volum_zxy
    ps_z, ps_x, ps_y = patch_zxy  # patch size z_xy
    pss_z, pss_x, pss_y = patch_step_zxy  # patch step z_xy
    assert pss_z < ps_z and pss_x < ps_x and pss_y < ps_y, \
        'invalid patch size/step param: step zxy:(%d, %d,%d) vs patch size:(%d, %d,%d)' % (
            pss_z, pss_x, pss_y, ps_z, ps_x, ps_y)
    coords = []

    z_axes_start, z_axex_end = 0, VZ
    if isinstance(specify_z_axes_range, (tuple, list)):
        z_axes_start, z_axex_end = specify_z_axes_range

    for cz in range(z_axes_start, z_axex_end, pss_z):
        for cx in range(0, VX, pss_x):
            for cy in range(0, VY, pss_y):
                coord = [
                    max(0, min(cz, VZ - ps_z)),
                    max(0, min(cx, VX - ps_x)),
                    max(0, min(cy, VY - ps_y)),
                ]
                coords += [coord]
                if VY - ps_y <= cy: break
            if VX - ps_x <= cx: break
        if VZ - ps_z <= cz: break

    return coords


def bbox3d_intersect(b1, b2):
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    if w1 <= 0 or h1 <= 0 or d1 <= 0 or w2 <= 0 or h2 <= 0 or d2 <= 0:
        return -1, -1, -1, 0, 0, 0
    x1_ = x1 + w1
    y1_ = y1 + h1
    z1_ = z1 + d1
    x2_ = x2 + w2
    y2_ = y2 + h2
    z2_ = z2 + d2
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    z3 = max(z1, z2)
    x3_ = min(x1_, x2_)
    y3_ = min(y1_, y2_)
    z3_ = min(z1_, z2_)
    if x3_ <= x3 or y3_ <= y3 or z3_ <= z3:
        return -1, -1, -1, 0, 0, 0
    return x3, y3, z3, x3_ - x3, y3_ - y3, z3_ - z3


def bbox3d_ratio(b1, b2):
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    xi, yi, zi, wi, hi, di = bbox3d_intersect(b1, b2)
    inter_volume = wi * hi * di * 1.0
    gt_volume = w2 * h2 * d2 * 1.0

    return inter_volume / gt_volume


def bbox3d_contain(b1, b2):
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2

    return x1 <= x2 and y1 <= y2 and z1 <= z2 and \
           (x1 + w1) >= (x2 + w2) and \
           (y1 + h1) >= (y2 + h2) and \
           (z1 + d1) >= (z2 + d2)
