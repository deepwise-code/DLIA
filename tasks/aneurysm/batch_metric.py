# -*-coding=utf-8 -*-
from functools import partial
import sys
import os
from skimage import measure
import nibabel as nib
import numpy as np
import cv2
import time

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def progress_monitor(total, verbose=1):
    '''状态参数'''
    finished_count = 0
    time_start = time.time()

    def _print(*args, ):
        nonlocal finished_count
        finished_count += 1
        if finished_count % verbose == 0:
            print("\rprogress: %d/%d = %.2f; time: %.2f s" % (
                finished_count, total, finished_count / total, time.time() - time_start), end="")
        if finished_count == total:
            print('\nfinished: %d, total cost time: %.2f s' % (total, time.time() - time_start,))

    return _print

def eval_segmentation(subject, seg_result, other_infos, mask_dir, blood_dir, use_blood, save_seg, save_dir):
    '''evaluate the segmentation of the case, with aneurysm mask and other info
    :param subject: patient id, type string
    :param seg_result: prediction matrix, type ndarray
    :param image_dir: path of image directory
    :param other_infos: dict, zxy_spacing or nii affine or ...
    :param mask_dir: path of aneurysm lesion mask
    :param blood_dir: vessel mask
    :param use_blood: evaluation condition, for reduce false positive not in blood vessel
    :param save_seg: flag for save the processed seg result
    :param save_dir: seg result save directory
    :return: lesion level recall and precision, and dice indicator of case seg and aneurysm mask
    '''
    '''load subject's correspoding aneurysm mask and vessel mask '''
    seg = seg_result  # the segmentation of corresponding image of specify patient id(symbol: subject)
    mask_path = os.path.join(mask_dir, '%s_mask.nii.gz' % subject)
    mask_path2 = os.path.join(mask_dir, '%s_aneurysm.nii.gz' % subject)
    if os.path.exists(mask_path):
        gt = nib.load(mask_path).get_data()  # (x, y, z) order
    elif os.path.exists(mask_path2):
        gt = nib.load(mask_path2).get_data()
    else:
        gt = np.zeros(seg.shape, 'uint8')  # for negative case, witch no aneurysm lesion mask
    if use_blood:
        blood = nib.load(os.path.join(blood_dir, '%s_blood.nii.gz' % subject)).get_data()
    else:
        blood = np.ones(seg.shape, 'uint8')
    if seg.shape != gt.shape or seg.shape != blood.shape:
        print(subject, 'error')

    # except for float dtype or others complex condition
    seg[seg > 0] = 1
    lesion_recall_seg = np.zeros(seg.shape, dtype=np.uint8)  # for calculate recall lession seg obj total dice
    lesion_recall_gt = np.zeros(seg.shape, dtype=np.uint8)
    lesion_recall_dice_list = []  # mask recall lesion level dice
    # gt[gt > 0] = 1

    '''2. before exec eval, set z axes range to filter segmentation lesion'''
    z_axes_height = 180  # type 1: experience value, 18CM
    z_spacing = other_infos['zxy_spacing'][0]
    z_axes_start = max(0, seg.shape[-1] - int(z_axes_height / z_spacing))  # type 1 strategy end
    # z_axes_start = max(0, int(seg.shape[-1] * 2 / 3 - 64))  # type 2: head-top, 1/3 + 64
    seg[..., :z_axes_start] = 0  # from neck(slice index 0) to target slice range, set zero

    '''3. seg lesion precision (accuracy) '''
    pred_inst = measure.label(seg)  # 分割结果3D联通实例
    props = measure.regionprops(pred_inst)
    lesion_precision_tp, lesion_precision_total = 0, 0
    for prop in props:
        x0, y0, z0, x1, y1, z1 = prop.bbox
        if prop.area >= 10 and (prop.image * blood[x0:x1, y0:y1, z0:z1]).sum() > 0:
            lesion_precision_total += 1
            if (prop.image * gt[x0:x1, y0:y1, z0:z1]).sum() > 0:
                lesion_precision_tp += 1
                lesion_recall_seg[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = prop.label  # tp seg lesion
        else:
            seg[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = 0

    if save_seg:
        assert save_dir is not None, 'invalid save directory: None'
        os.makedirs(save_dir, exist_ok=True)
        affine = other_infos['affine']
        save_path = '%s/%s_seg.nii.gz' % (save_dir, subject)
        seg_img = nib.Nifti1Image(seg, affine)
        nib.save(seg_img, save_path)

    '''4. mask lesion recall  (accuracy) '''
    mask_inst = measure.label(gt)
    props = measure.regionprops(mask_inst)
    lesion_recall_total, lesion_recall_tp = len(props), 0
    for prop in props:
        x0, y0, z0, x1, y1, z1 = prop.bbox
        if (prop.image * seg[x0:x1, y0:y1, z0:z1]).sum() > 0:  # GT区域与分割结果区域有交集,则召回
            lesion_recall_tp += 1
            # label recall mask and calculate lesion level dice
            lesion_recall_gt[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = prop.label  # mask recall lesion
            seg_lesion_labels = np.unique(lesion_recall_seg[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]])
            intersect = (prop.image * seg[x0:x1, y0:y1, z0:z1]).sum()
            tp_lesions_voxel = sum(
                (lesion_recall_seg == lesion_label).sum() for lesion_label in seg_lesion_labels if lesion_label > 0)
            mask_lesion_voxel = prop.area  # EQ: len(prop.coords); coords list witch item (x,y,z ...) order
            lesion_recall_dice = 2.0 * intersect / (tp_lesions_voxel + mask_lesion_voxel)
            lesion_recall_dice_list += [lesion_recall_dice]

    # dice indicator calculate
    dice = -1  # default value for negative case
    case_true_positive_recall_dice = -1
    if gt.sum() > 0:  # for aneurysm positive case, dice >= 0; in particular, 0 is no any lesion recall
        gt[gt > 0] = 1
        overlap = (seg * gt).sum()
        dice = 2.0 * overlap / (seg.sum() + gt.sum())  # all voxel in lesion pred and all voxel in aneurysm mask

        lesion_recall_seg[lesion_recall_seg > 0] = 1
        intersect = (lesion_recall_seg * gt).sum()
        case_true_positive_recall_dice = 2.0 * intersect / (lesion_recall_seg.sum() + gt.sum())

    return dice, case_true_positive_recall_dice, lesion_recall_dice_list, \
           lesion_recall_tp, lesion_recall_total, lesion_precision_tp, lesion_precision_total


def evaluation(eval_results, image_dir, mask_dir, blood_dir='', use_blood=False, save_seg=False, save_dir=None):
    ''' evaluate aneurysm segmentation results in multi processing for speed up
    :param eval_results: dict, and the key is patient id and value is tuple of segmentation ndarray and other infos
    :param image_dir: when save seg result, find corresponding image to extract affine info
    :param mask_dir: aneurysm lesion mask store directory
    :param blood_dir: vessel segmentation mask directory
    :param use_blood: evaluate aneurysm use vessel mask for reduce false positive lesion
    :param save_seg: save seg result to pid.nii.gz to specify directory
    :param save_dir: specify the seg result save directory
    :return: None
    '''
    subjects = []  # patient id list
    seg_results = []  # segmentation result, type: ndarray, dim order: (z, x, y)
    other_infos = []
    eval_one = partial(eval_segmentation, mask_dir=mask_dir, blood_dir=blood_dir,
                       use_blood=use_blood, save_seg=save_seg, save_dir=save_dir)
    monitor = progress_monitor(total=len(eval_results))
    with ProcessPoolExecutor(max_workers=16) as ex:
        rets = []
        for subject, (seg_result, _other_infos) in eval_results.items():
            subjects += [subject]
            seg_results += [seg_result]
            other_infos += [_other_infos]
            future = ex.submit(eval_one, subject, seg_result, _other_infos)
            future.add_done_callback(fn=monitor)
            rets += [future]
        print('eval data in async ...')
        ex.shutdown(wait=True)
        results = [ret.result() for ret in rets]
        # results = ex.map(eval_one, subjects, seg_results, other_infos)

    # results = np.array(list(results), dtype='float32')
    results = list(results)

    # calculate three type dice indicator
    dice_list = []  # all positive case's mean dice, witch the case dice calc with all pos-pred & all GT voxel
    case_true_positive_recall_dice_list = []  # all positive case's mean dice, witch the case dice calc with only TP lesion & all gt
    lesion_recall_dice_list = []  # lesion level's mean dice, only include the recall lesion's dice in GT
    # recall, precision, sensitivity, specificity
    lesion_recall_tp, lesion_recall_total = 0, 0  # lesion level
    lesion_precision_tp, lesion_precision_total = 0, 0
    person_positive_tp, person_positive_total = 0, 0  # person level
    person_negative_tp, person_negative_total = 0, 0
    fp_num = 0
    for (subject, result) in zip(subjects, results):  # PID + 对应组评测结果

        _dice, _case_true_positive_recall_dice, _lesion_recall_dice_list, \
        _lesion_recall_tp, _lesion_recall_total, _lesion_precision_tp, _lesion_precision_total = result

        lesion_recall_dice_list.extend(_lesion_recall_dice_list)

        if _lesion_recall_total > 0:  # for person positive case
            dice_list.append(_dice)
            case_true_positive_recall_dice_list += [_case_true_positive_recall_dice]
            person_positive_total += 1

        if _lesion_recall_tp > 0:  # for recall positive person
            person_positive_tp += 1

        lesion_recall_tp += _lesion_recall_tp
        lesion_recall_total += _lesion_recall_total

        lesion_precision_tp += _lesion_precision_tp
        lesion_precision_total += _lesion_precision_total

        if _lesion_recall_total == 0:  # for person negative case
            person_negative_total += 1
            if _lesion_precision_total == 0:
                person_negative_tp += 1

        _fp_num = _lesion_precision_total - _lesion_precision_tp
        fp_num += _fp_num
        _lesion_recall_dice_avg = np.mean(_lesion_recall_dice_list) if len(_lesion_recall_dice_list) > 0 else 0.

        # 打印组PID, 病灶召回,准确率; 误检个数; Dice
        print('%s: Recall=%d/%d, Precision=%d/%d, FP=%d, Dice=%.4f, CaseTPLesionDICE=%.4f, LesionRecallDice=%.4f' % (
            subject, _lesion_recall_tp, _lesion_recall_total, _lesion_precision_tp, _lesion_precision_total, _fp_num,
            _dice, _case_true_positive_recall_dice, _lesion_recall_dice_avg))

    # result summary
    recall = lesion_recall_tp / (lesion_recall_total + 1e-6)
    precision = lesion_precision_tp / (lesion_precision_total + 1e-6)
    sensitivity = person_positive_tp / (person_positive_total + 1e-6)
    specificity = person_negative_tp / (person_negative_total + 1e-6)
    case_dice = np.mean(dice_list) if len(dice_list) > 0 else 0.
    case_true_positive_recall_dice = np.mean(case_true_positive_recall_dice_list) \
        if len(case_true_positive_recall_dice_list) > 0 else 0.
    lesion_recall_dice = np.mean(lesion_recall_dice_list) if len(lesion_recall_dice_list) > 0 else 0.
    print('lesion-level: precision=%.4f(%d/%d),recall=%.4f(%d/%d)' % (
        precision, lesion_precision_tp, lesion_precision_total,
        recall, lesion_recall_tp, lesion_recall_total,))
    print('person-level: Sensitivity=%.4f(%d/%d),Specificity=%.4f(%d/%d)' % (
        sensitivity, person_positive_tp, person_positive_total,
        specificity, person_negative_tp, person_negative_total))
    print('Dice=%.4f,CaseTPLesionDICE=%.4f,LesionRecallDice=%.4f' % (
        case_dice, case_true_positive_recall_dice, lesion_recall_dice))
    TEMPLATE = 'PRECISION=%.4f(%d/%d);RECALL=%.4f(%d/%d);Sensitivity=%.4f(%d/%d);Specificity=%.4f(%d/%d);' \
               'Dice=%.4f;CaseTPLesionDICE=%.4f;LesionRecallDice=%.4f'
    return TEMPLATE % (
        precision, lesion_precision_tp, lesion_precision_total,  # lesion level, lesion pred accuracy
        recall, lesion_recall_tp, lesion_recall_total,  # lesion level, lesion pred recall
        sensitivity, person_positive_tp, person_positive_total,  # patient level, diagnose positive patient accuracy
        specificity, person_negative_tp, person_negative_total,  # patient level, diagnose healthy accuracy
        case_dice, case_true_positive_recall_dice, lesion_recall_dice,
    )


if __name__ == '__main__':
    pass
    # seg_files = os.listdir(SEG_FOLDER)
    # seg_files = list(filter(lambda x: '.nii.gz' in x, seg_files))
    # evaluation(seg_files)
