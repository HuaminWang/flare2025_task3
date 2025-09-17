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
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from models.models import create_model
from data.CT_MRI_unaligned_dataset import CT_MRI_UnalignedDataset
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import numpy as np
import cv2

def merge_multiple_img(vis_result, imtype=np.uint8):
    img_num = len(vis_result.keys())
    col_num = int(max(round(img_num ** 0.5), 1))
    row_num = int(np.ceil(img_num / col_num))
    h, w, c = list(vis_result.values())[0].shape
    whole_img = np.zeros([h * row_num, w * col_num, c], dtype=imtype)
    index = 0
    for k, v in vis_result.items():
        this_row = int(index / col_num)
        this_col = int(index % col_num)
        img = v.copy()
        cv2.putText(img, k, (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        whole_img[this_row * h: this_row * h + h, this_col * w: this_col * w + w] = img
        index += 1
    return whole_img.astype(imtype)

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'tensorboard_exp'))
    total_steps = 0
    print('Loading data......')

    train_dset = CT_MRI_UnalignedDataset()
    train_dset.initialize(opt)
    print('train set:{}'.format(len(train_dset)))

    train_loader = DataLoader(train_dset, batch_size=opt.batchSize, shuffle=True, pin_memory=False, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i_iter, data in enumerate(train_loader):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data)
            if i_iter % opt.N_disc == 0:
                model.optimize_parameters(True)
            else:
                model.optimize_parameters(False)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visual_results = model.get_current_visuals()
                visualizer.display_current_results(visual_results, epoch, save_result)
                if save_result:
                    vis_merge = merge_multiple_img(visual_results)
                    writer.add_image('visualization', vis_merge, global_step=total_steps, dataformats='HWC')

                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize

                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

                for k, v in errors.items():
                    writer.add_scalar(k, v, global_step=total_steps)


        if epoch % opt.save_epoch_freq == 0:
            model.save(epoch)
        model.update_learning_rate()

    writer.close()