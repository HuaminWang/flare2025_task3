import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import random
import torch.nn.functional as F

class CT_MRI_UnalignedDataset_per_case(Dataset):
    def __init__(self, case_dir):
        self.files = [os.path.join(case_dir, file) for file in os.listdir(case_dir) if '.npy' in file]
        self.files = sorted(self.files, key=lambda x: int(os.path.basename(x).split('.npy')[0].split('_')[-1]))
        self.if_CT = True if 'CT' in case_dir.split('/')[-2] else False

    def __getitem__(self, index):
        img = np.load(self.files[index])
        if self.if_CT:
            img = img.clip(-175.0, 350.)
            img = (img + 175.) / (350. + 175.)
            img = img * 2.0 - 1.0
            key = 'CT'
        else:
            key = 'MR'
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
