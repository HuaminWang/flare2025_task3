import numpy as np
import SimpleITK as sitk
from models.Organ_attention import Organ_attention
from dataset import CTDataset
import os
from glob import glob
import torch
from torch.utils.data import DataLoader
import traceback
from options.test_options import TestOptions
from scipy.ndimage import zoom


def maymkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def convert_to_origin_size(volume, pad_info, origin_size):
    scale = origin_size[1] / 256.0
    volume = zoom(volume, zoom=[1, scale, scale], order=1)
    volume = volume[:, pad_info[0]: origin_size[1] - pad_info[1], pad_info[2]: origin_size[2] - pad_info[3]]
    return volume

def MRI2CT(file, model, outfolder):
    simage = sitk.ReadImage(file)
    dset = CTDataset(simage)
    dloader = DataLoader(dset, batch_size=8, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    fake_MRIs = []
    with torch.no_grad():
        for batch_no, batch_img in enumerate(dloader):
            model.set_test_input(batch_img)
            result = model.test_forward()
            fake_mri = result['fake_B']
            fake_mri = fake_mri.squeeze(1).cpu().numpy()
            fake_MRIs.append(fake_mri)

    fake_MRIs = np.concatenate(fake_MRIs, axis=0)
    pad_info = dset._get_pad_info()
    origin_size = dset._get_origin_size()
    fake_MRIs = convert_to_origin_size(fake_MRIs, pad_info, origin_size)
    sitk_fake_mri = sitk.GetImageFromArray(fake_MRIs)
    sitk_fake_mri.CopyInformation(simage)

    file_name = os.path.basename(file).split('.nii')[0]

    sitk.WriteImage(sitk_fake_mri, os.path.join(outfolder, file_name+'_mri.nii.gz'))

def batch_MRI2CT(dataroot, model, outfolder=None):
    if not outfolder:
        outfolder = os.path.join(os.path.dirname(dataroot), 'fake_mri')
    maymkdir(outfolder)

    files = sorted(glob(os.path.join(dataroot, '*.nii.gz')))
    for f_no, file in enumerate(files):
        try:
            MRI2CT(file, model, outfolder)
        except:
            traceback.print_exc()

def main():
    dataroot = '/data/MultiOrganSeg/code/stargan-v2-master/data/nii_images/test'
    outfolder = '/data/MultiOrganSeg/code/stargan-v2-master/data/nii_images/test/fake_MRI'
    opt = TestOptions().parse()

    model = Organ_attention()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    model.net_G_A_load_weight(opt.which_epoch)
    model.netG_A.eval()

    batch_MRI2CT(dataroot, model, outfolder)

if __name__ == '__main__':
    main()



