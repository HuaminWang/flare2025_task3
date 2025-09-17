import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import SimpleITK as sitk
from scipy.ndimage import zoom
import torch.nn.functional as F

class CTDataset(Dataset):
    def __init__(self, simage):
        win_min, win_max = -175, 350
        self.volume = sitk.GetArrayFromImage(simage)
        self.volume = self.volume.clip(win_min, win_max)

        origin_w, origin_h = simage.GetSize()[:-1]
        t = max(origin_h, origin_w)
        h_pad = t - origin_h
        if h_pad != 0:
            h_pad_low = int(h_pad // 2)
            h_pad_high = h_pad - h_pad_low
        else:
            h_pad_low = 0
            h_pad_high = 0

        w_pad = t - origin_w
        if w_pad != 0:
            w_pad_low = int(w_pad // 2)
            w_pad_high = w_pad - w_pad_low
        else:
            w_pad_low = 0
            w_pad_high = 0

        scale = 256.0 / t
        self.volume = np.pad(self.volume, ((0, 0), (h_pad_low, h_pad_high), (w_pad_low, w_pad_high)), mode='constant',
                             constant_values=win_min)

        self.origin_size = self.volume.shape

        self.volume = (self.volume - win_min) / (win_max - win_min)
        self.volume = self.volume * 2.0 - 1.0
        self.volume = zoom(self.volume, zoom=[1, scale, scale], order=1)

        self.pad_info = [h_pad_low, h_pad_high, w_pad_low, w_pad_high]

    def _get_pad_info(self):
        return self.pad_info

    def _get_origin_size(self):
        return self.origin_size

    def __getitem__(self, index):
        img = self.volume[index]
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)

        return {'CT': img}

    def __len__(self):
        return len(self.volume)