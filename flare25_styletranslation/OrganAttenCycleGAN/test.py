# -*- coding: utf-8 -*-
"""
Created on Jan 05 2019
@author: Jue Jiang
Date modified: July 07 2020
@author: Harini Veeraraghavan

Wrapper code for PSIGAN segmentor training

"""
# -*- coding: utf-8 -*-

import time
from options.test_options import TestOptions
from models.models import create_model
from data.CT_MRI_unaligned_dataset_per_case import CT_MRI_UnalignedDataset_per_case
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
import SimpleITK as sitk
from scipy.ndimage import zoom

save_nii=True

def resize_and_save_with_scipy(image, ref_img):
    zoom_factors = (ref_img.shape[0] / image.shape[0], ref_img.shape[1] / image.shape[1], ref_img.shape[2] / image.shape[2] )
    resized_image = zoom(image, zoom_factors, order=1)
    return resized_image

def to_png(arr):
    arr = arr / 8.0 * 255.0
    print(arr.shape,'28')
    if len(arr.shape) == 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
    if arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    return arr.astype(np.uint8)

if __name__ == '__main__':
    opt = TestOptions().parse()
    model = create_model(opt)
    if opt.G_A:
        
        if opt.reverse_test:
            data_dir = os.path.join(opt.dataroot, opt.phase, 'img_aug')
            # ref_root_a = '/data2/jianghao/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task012_flare/PublicValidation/imagesVal'
            ref_root_a = '/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/validation/PET_imagesVal/'
        else:
            data_dir = os.path.join(opt.dataroot, opt.phase, 'img')
            # ref_root_a = '/data2/jianghao/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task012_flare/CT/images'
            ref_root_a = '/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/train_CT_gt_label/imagesTr'
        model.net_G_A_load_weight(opt.which_epoch)
        model.netG_A.eval()
    if opt.G_B:
        data_dir = os.path.join(opt.dataroot, opt.phase, 'img_aug')
        # ref_root_b = '/data2/jianghao/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task012_flare/PublicValidation/imagesVal'
        ref_root_b = '/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/validation/PET_imagesVal/'
        model.net_G_B_load_weight(opt.which_epoch)
        model.netG_B.eval()

    if opt.Seg:
        data_dir = os.path.join(opt.dataroot, opt.phase, 'img_aug')
        model.net_Seg_B_load_weight(opt.which_epoch)
        model.netSeg.eval()

    cases = [os.path.join(data_dir, case) for case in os.listdir(data_dir)]
    cases.sort()
    for case_no, case in enumerate(cases):
        print(case_no, case)
        test_dset = CT_MRI_UnalignedDataset_per_case(case, opt.reverse_test)
        test_loader = DataLoader(test_dset, batch_size=opt.batchSize, shuffle=False, pin_memory=False, num_workers=8)

        if opt.G_A:
            fake_Bs = []
        if opt.G_B:
            fake_As = []
        if opt.Seg:
            real_B_segs = []
        slices = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.set_test_input(data)
                result = model.test_forward()
                if 'fake_B' in result:
                    fake_B = result['fake_B']
                    fake_B = fake_B.squeeze(1).cpu().numpy()
                    fake_Bs.append(fake_B)
                if 'fake_A' in result:
                    fake_A = result['fake_A']
                    fake_A = fake_A.squeeze(1).cpu().numpy()
                    fake_As.append(fake_A)
                if 'real_B_seg' in result:
                    real_B_seg = result['real_B_seg'].cpu().numpy()
                    real_B_segs.append(real_B_seg)
                slices.extend(list(data['slice']))

        if opt.G_A:
            fake_Bs = np.concatenate(fake_Bs, axis=0)
            if not save_nii:
                save_case = os.path.join(opt.results_dir, 'fake_B', os.path.basename(case))
                if not os.path.exists(save_case):
                    os.makedirs(save_case)
                else:
                    continue
                for s_no in range(len(fake_Bs)):
                    np.save(os.path.join(save_case, slices[s_no]), fake_Bs[s_no])
            else:
                save_file = os.path.join(opt.results_dir, 'fake_B')
                if not os.path.exists(save_file):
                    os.makedirs(save_file)
                
                ref_image_a = sitk.ReadImage(os.path.join(ref_root_a, os.path.basename(case) + '_0000.nii.gz'))
                ref_arr = sitk.GetArrayFromImage(ref_image_a)
                sitk_fake_B = resize_and_save_with_scipy(fake_Bs, ref_arr)
                sitk_fake_B = sitk.GetImageFromArray(sitk_fake_B)
                print(ref_image_a,'101')
                sitk_fake_B.CopyInformation(ref_image_a)
                sitk.WriteImage(sitk_fake_B, os.path.join(save_file, opt.target_domian+'_'+os.path.basename(case) + '.nii.gz'))
        if opt.G_B:
            fake_As = np.concatenate(fake_As, axis=0)
            if not save_nii:
                save_case = os.path.join(opt.results_dir, 'fake_A', os.path.basename(case))
                if not os.path.exists(save_case):
                    os.makedirs(save_case)
                else:
                    continue
                for s_no in range(len(fake_As)):
                    # np.save(os.path.join(save_case, slices[s_no]), fake_As[s_no])
                    print(fake_As[s_no].shape)
                    cv2.imwrite(os.path.join(save_case, '{}.png'.format(slices[s_no])), to_png(fake_As[s_no]))
            else:
                save_file = os.path.join(opt.results_dir, 'fake_A')
                if not os.path.exists(save_file):
                    os.makedirs(save_file)
                
                ref_image_a = sitk.ReadImage(os.path.join(ref_root_b, os.path.basename(case) + '_0000.nii.gz'))
                ref_arr = sitk.GetArrayFromImage(ref_image_a)
                sitk_fake_A = resize_and_save_with_scipy(fake_As, ref_arr)
                sitk_fake_A = sitk.GetImageFromArray(sitk_fake_A)
                print(ref_image_a,'102')
                sitk_fake_A.CopyInformation(ref_image_a)
                sitk.WriteImage(sitk_fake_A, os.path.join(save_file, opt.target_domian+'_'+os.path.basename(case) + '.nii.gz'))
                

        if opt.Seg:
            real_B_segs = np.concatenate(real_B_segs, axis=0).astype(np.uint8)
            if not save_nii:
                save_case = os.path.join(opt.results_dir, 'real_B_seg', os.path.basename(case))
                if not os.path.exists(save_case):
                    os.makedirs(save_case)
                else:
                    continue
                for s_no in range(len(real_B_segs)):
                    np.save(os.path.join(save_case, slices[s_no]), real_B_segs[s_no])
                    # cv2.imwrite(os.path.join(save_case, '{}.png'.format(slices[s_no])), to_png(real_B_segs[s_no]))
            else:
                save_file = os.path.join(opt.results_dir, 'real_B_seg')
                if not os.path.exists(save_file):
                    os.makedirs(save_file)
                sitk_real_B_segs = sitk.GetImageFromArray(real_B_segs)
                sitk.WriteImage(sitk_real_B_segs, os.path.join(save_file, os.path.basename(case) + '.nii.gz'))