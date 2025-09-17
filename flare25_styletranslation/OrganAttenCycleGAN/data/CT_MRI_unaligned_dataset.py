import numpy as np
import os
from .base_dataset import BaseDataset
from torch.utils.data import DataLoader
import torch
import random
import torch.nn.functional as F
import pandas as pd

import torch
import torch.nn.functional as F
import random
import numpy as np

def transform(tensor, opt):
    #=======resize and pad=======
    original_height = tensor.size(2)
    original_width = tensor.size(3)
    aspect_ratio = original_height / original_width
    

    if aspect_ratio > 1:  # Height is greater than width
        new_height = opt.loadSize
        new_width = int(new_height / aspect_ratio)
    else:  # Width is greater than height
        new_width = opt.loadSize
        new_height = int(new_width * aspect_ratio)

    # Resize
    if tensor.size(1) == 2:
        tensor1 = F.interpolate(tensor[:, :1], size=(new_height, new_width), mode='bilinear', align_corners=False)
        tensor2 = F.interpolate(tensor[:, 1:], size=(new_height, new_width), mode='nearest')
        tensor = torch.cat([tensor1, tensor2], dim=1)
    elif tensor.size(1) == 1:
        tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # Pad
    pad_height = (opt.loadSize - new_height) // 2
    pad_width = (opt.loadSize - new_width) // 2
    tensor = F.pad(tensor, (pad_width, pad_width, pad_height, pad_height), 'constant', -1)

    # Random crop
    tensor = random_crop(tensor, [opt.fineSize, opt.fineSize])

    #=========horizontal flip=========
    if random.randint(0, 2):
        w_range = list(np.arange(tensor.size(-1)))[::-1]
        tensor = tensor[:, :, :, w_range]

    return tensor

# def random_crop(tensor, size):
#     # This is a placeholder function for random cropping
#     # Assume tensor is already padded appropriately
#     # Crop logic would go here
#     return tensor


# def transform(tensor, opt):
#     #=======resize and crop=======
#     if random.randint(1, 2):
#         osize = [opt.loadSize, opt.loadSize]
#         if tensor.size(1) == 2:
#             tensor1 = F.interpolate(tensor[:, :1], size=osize, mode='bilinear')
#             tensor2 = F.interpolate(tensor[:, 1:], size=osize, mode='nearest')
#             tensor = torch.cat([tensor1, tensor2], dim=1)
#         elif tensor.size(1) == 1:
#             tensor = F.interpolate(tensor, size=osize, mode='bilinear')

#         tensor = random_crop(tensor, [opt.fineSize, opt.fineSize])

#     #=========horizontal flip=========
#     if random.randint(0, 2):
#         w_range = list(np.arange(tensor.size(-1)))[::-1]
#         tensor = tensor[:, :, :, w_range]

#     return tensor

def random_crop(tensor, output_size, pad_value=-1):
    in_h, in_w = tensor.size()[-2:]
    out_h, out_w = output_size

    pad_flag = False
    if in_h < out_h:
        h_pad = out_h - in_h
        h_pad_low = int(h_pad // 2)
        h_pad_high = h_pad - h_pad_low
        pad_flag = True
    else:
        h_pad_low, h_pad_high = 0, 0
    if in_w < out_w:
        w_pad = out_w - in_w
        w_pad_low = int(w_pad // 2)
        w_pad_high = w_pad - w_pad_low
        pad_flag = True
    else:
        w_pad_low, w_pad_h = 0, 0

    if pad_flag:
        tensor = F.pad(tensor, pad=(w_pad_low, w_pad_high, h_pad_low, h_pad_high), value=pad_value)

    in_h, in_w = tensor.size()[-2:]
    if in_h == out_h and in_w == out_w:
        return tensor
    else:
        h, w = out_h, out_w
        i = random.randint(0, in_h - out_h)
        j = random.randint(0, in_w - out_h)
        return tensor[:, :, i:i+h, j:j+w]

class CT_MRI_UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.target_domian = opt.target_domian 
        self.opt = opt
        self.data_dir = opt.dataroot
        reverse_test = opt.reverse_test
        if reverse_test:
            self.dir_CT_imgs = os.path.join(self.data_dir, opt.phase, self.target_domian)
            self.dir_MRI_imgs = os.path.join(self.data_dir, opt.phase, 'img')
            if opt.partial_label:
                self.dir_CT_labels = os.path.join(self.data_dir, opt.phase, 'lab_aug')
            else:
                self.dir_CT_labels = os.path.join(self.data_dir, opt.phase, 'lab_aug')

            if opt.phase == 'test':
                self.dir_MRI_labels = os.path.join(self.data_dir, opt.phase, 'img')
            else:
                self.dir_MRI_labels = None
        else:   
            self.dir_CT_imgs = os.path.join(self.data_dir, opt.phase, 'img')
            self.dir_MRI_imgs = os.path.join(self.data_dir, opt.phase, self.target_domian)
            if opt.partial_label:
                self.dir_CT_labels = os.path.join(self.data_dir, opt.phase, 'lab')
            else:
                self.dir_CT_labels = os.path.join(self.data_dir, opt.phase, 'lab')

            if opt.phase == 'test':
                self.dir_MRI_labels = os.path.join(self.data_dir, opt.phase, self.target_domian)
            else:
                self.dir_MRI_labels = None

        self.CT_imgs = sorted([os.path.join(self.dir_CT_imgs, case) for case in os.listdir(self.dir_CT_imgs)])
        self.MRI_imgs = sorted([os.path.join(self.dir_MRI_imgs, case) for case in os.listdir(self.dir_MRI_imgs)])

        if opt.partial_label:
            csv_file = os.path.join(self.data_dir, 'partial_label_info.csv')
            df = pd.read_csv(csv_file)
            df = df[df.subset == 0] if opt.phase == 'train' else df[df.subset == 1]
            CT_caseids = [os.path.basename(case) for case in self.CT_imgs]
            self.partial_label_dict = {}
            for caseid in CT_caseids:
                this_row = df[df.final_uid == caseid+'_0000']
                if len(this_row) == 1:
                    self.partial_label_dict[this_row['final_uid'].values[0][:-5]] = this_row.values[0][5:]
                else:
                    print(caseid)
        else:
            self.partial_label_dict = None

        self.CT_size = len(self.CT_imgs)
        self.MRI_size = len(self.MRI_imgs)

    def __getitem__(self, index):
        index_CT = index % self.CT_size
        CT_img = self.CT_imgs[index_CT]
        if self.opt.serial_batches:
            index_MRI = index % self.MR_size
        else:
            index_MRI = np.random.randint(0, self.MRI_size)
        MRI_img = self.MRI_imgs[index_MRI]

        #=========image loading===========
        CT_slices = os.listdir(CT_img)
        MRI_slices = os.listdir(MRI_img)

        CT_slice_index = np.random.randint(len(CT_slices))
        MRI_slice_index = np.random.randint(len(MRI_slices))

        CT_slice_name = CT_slices[CT_slice_index]
        MRI_slice_name = MRI_slices[MRI_slice_index]

        CT_slice_img = np.load(os.path.join(CT_img, CT_slice_name))
        MRI_slice_img = np.load(os.path.join(MRI_img, MRI_slice_name))
        MRI_slice_img = np.flip(MRI_slice_img, axis=0)
        MRI_slice_img = np.ascontiguousarray(MRI_slice_img)
        # print(MRI_slice_img.shape)

        CT_slice_label = np.load(os.path.join(self.dir_CT_labels, os.path.basename(CT_img), CT_slice_name))
        if self.dir_MRI_labels is not None:
            MRI_slice_label = np.load(os.path.join(self.dir_MRI_labels, os.path.basename(MRI_img), MRI_slice_name))

        #=========normalization==========

        CT_slice_img = torch.from_numpy(CT_slice_img).float()
        CT_slice_label = torch.from_numpy(CT_slice_label.astype(np.uint8)).float()
        # print(CT_slice_label.max(), CT_slice_label.min(), '191')
        CT_slice = torch.stack([CT_slice_img, CT_slice_label]).unsqueeze(0)

        MRI_slice_img= torch.from_numpy(MRI_slice_img).float()
        if self.dir_MRI_labels is not None:
            MRI_slice_label = torch.from_numpy(MRI_slice_label).float()
            MRI_slice = torch.stack([MRI_slice_img, MRI_slice_label]).unsqueeze(0)
        else:
            MRI_slice = MRI_slice_img.unsqueeze(0).unsqueeze(0)
        if self.opt.data_aug:
            CT_slice = transform(CT_slice, self.opt)
            MRI_slice = transform(MRI_slice, self.opt)
        #==============partial label==================
        if self.partial_label_dict is not None:
            CT_partial_label = self.partial_label_dict[os.path.basename(CT_img)]
            if np.sum(CT_partial_label) == len(CT_partial_label):
                CT_partial_label = [1,] + list(CT_partial_label)
            else:
                CT_partial_label = [0,] + list(CT_partial_label)
            CT_partial_label = torch.from_numpy(np.array(CT_partial_label)).float()
            return {'CT': CT_slice.squeeze(0), 'MRI': MRI_slice.squeeze(0), 'CT_partial_label': CT_partial_label}
        else:
            return {'CT': CT_slice.squeeze(0), 'MRI': MRI_slice.squeeze(0)}


    def __len__(self):
        return max(self.CT_size, self.MRI_size) * 30

    def name(self):
        return "CT_MRI_UnalignedDataset"


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    parse = argparse.ArgumentParser('CycleGan with Organ Attention')
    parse.add_argument('--dataroot', type=str, default='/data/MultiOrganSeg/MR_data/npy', help='path to dataset')
    parse.add_argument('--phase', type=str, default='train', help='train or test')
    parse.add_argument('--loadSize', type=int, default=300, help='scale size')
    parse.add_argument('--fineSize', type=int, default=256, help='input size')
    parse.add_argument('--serial_batches', type=bool, default=False, help='whether to use serialized batch')
    parse.add_argument('--batch_size', type=int, default=2, help='batch size')
    parse.add_argument('--data_aug', type=bool, default=True, help='whether to use data augmentation')

    args = parse.parse_args()

    dset = CT_MRI_UnalignedDataset()
    dset.initialize(args)
    print('Dataset: {}'.format(dset.name()))
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=4)

    for batch_no, batch in enumerate(loader):
        CT = batch['CT'].numpy()
        MR = batch['MRI'].numpy()
        for i in range(len(CT)):
            fig = plt.figure(i+1)
            this_CT = CT[i]
            this_MR = MR[i]

            plt.subplot(221)
            plt.imshow(this_CT[0], cmap='gray')
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(this_CT[1], cmap='gray')
            plt.axis('off')
            plt.subplot(223)
            plt.imshow(this_MR[0], cmap='gray')
            plt.axis('off')

            if len(this_MR) > 1:
                plt.subplot(224)
                plt.imshow(this_MR[1], cmap='gray')
                plt.axis('off')
            plt.show()