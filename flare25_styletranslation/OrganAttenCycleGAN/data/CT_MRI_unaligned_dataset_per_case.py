import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import random
import torch.nn.functional as F
from scipy.ndimage import zoom

def resize_with_scipy(image, target_size=(320, 320)):
    # 计算缩放因子
    zoom_factors = (target_size[0] / image.shape[0], target_size[1] / image.shape[1])
    
    # 调整大小
    resized_image = zoom(image, zoom_factors, order=1)  # 使用线性插值
    return resized_image



class CT_MRI_UnalignedDataset_per_case(Dataset):
    def __init__(self, case_dir, reverse_test):
        self.files = [os.path.join(case_dir, file) for file in os.listdir(case_dir) if '.npy' in file]
        self.files = sorted(self.files, key=lambda x: int(os.path.basename(x).split('.npy')[0].split('_')[-1]))
        if reverse_test:
            self.if_CT = True if 'amos' in case_dir.split('/')[-1] else False
        else:
            self.if_CT = True if 'FLARE' in case_dir.split('/')[-1] else False

    def __getitem__(self, index):
        img = np.load(self.files[index])
        if self.if_CT:
            key = 'CT'
        else:
            key = 'MR'
        img = resize_with_scipy(img)
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        return {key: img, 'slice': os.path.basename(self.files[index]).split('.')[0]}

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    case_dir = '/data/MultiOrganSeg/MR_data/npy/train/CT_images/multiorgan_00001'
    dset = CT_MRI_UnalignedDataset_per_case(case_dir)
    loader = DataLoader(dset, batch_size=2, shuffle=False, pin_memory=False, num_workers=4)

    for b_no, batch in enumerate(loader):
        if 'CT' in batch.keys():
            img = batch['CT']
        elif 'MR' in batch.keys():
            img = batch['MR']
        print(batch['slice'])
        print(img.size())
        print()
