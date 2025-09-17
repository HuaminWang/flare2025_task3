import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2

def rename_files_in_directory(directory, old_rep, new_rep, to_0000 = False, to_0001 = None, prefix = None ):
    for filename in os.listdir(directory):
        if old_rep in filename:
            old_file_path = os.path.join(directory, filename)
            new_filename = filename.replace(old_rep, new_rep)
            
            if to_0000:
                if not '0000' in new_filename:
                    new_filename = new_filename.replace('.nii.gz', '_0000.nii.gz')
            if to_0001 is not None:
                if '0000' in new_filename:
                    new_filename = new_filename.replace('_0000', to_0001)
            if prefix is not None:
                new_file_path = os.path.join(directory, prefix + new_filename)
            else:
                new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} to {new_file_path}")

def to_npy(image_files, image_folder_path, output_image_folder, isimage = True, isCT = False, isMRI = False,save_png = False):
    for image_file in sorted(image_files):
        image_path = os.path.join(image_folder_path, image_file)
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)

        if isimage:
            if isCT:
                image[image>600] = 600
                image[image<-600] = -600
            elif isMRI:
                lower_percentile = np.percentile(image, 0.1)
                upper_percentile = np.percentile(image, 99.9)
                image = np.clip(image, lower_percentile, upper_percentile)
                print(image_file, lower_percentile, upper_percentile)
            else:
                image = image
            print(image.max(), image.min())
            image = (image - image.min()) / (image.max() - image.min())
            image = image * 2.0 - 1.0

        image_base_name = os.path.splitext(os.path.splitext(image_file)[0])[0]
        if '0000' in image_base_name:
            image_base_name = image_base_name.replace('_0000', '')
        image_output_dir = os.path.join(output_image_folder, image_base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        print(image.shape,image_file )
        for i in range(image.shape[0]):
            image_slice = image[i, :, :]
            image_slice_path = os.path.join(image_output_dir, f'{image_base_name}_slice_{i}.npy')
            # if not os.path.exists(image_slice_path):
            print(image.min(), image.max())
            np.save(image_slice_path, image_slice)
            if save_png:
                png_slice_path = os.path.join(image_output_dir, f'{image_base_name}_slice_{i}.png')
                cv2.imwrite(png_slice_path, image_slice)

def norm_test(image_files, image_folder_path, save_path, isCT = False, isMRI = False, prefix = None):
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image = sitk.ReadImage(image_path)
        image_arr = sitk.GetArrayFromImage(image)
        dd,ww,hh = image_arr.shape
        for i in range(dd):
            if image_arr[i].sum() == 0:
                print(image_file)

def norm(image_files, image_folder_path, save_path, isCT = False, isMRI = False, prefix = None):
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image = sitk.ReadImage(image_path)
        image_arr = sitk.GetArrayFromImage(image)
        if isCT:
                image_arr[image_arr>120] = 120
                image_arr[image_arr<-20] = -20
        elif isMRI:
            lower_percentile = np.percentile(image_arr, 1)
            upper_percentile = np.percentile(image_arr, 99)
            
            image_arr = np.clip(image_arr, lower_percentile, upper_percentile)
        else:
            image_arr = image_arr
        
        image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min())
        image_arr = image_arr * 2.0 - 1.0

        image_arr = sitk.GetImageFromArray(image_arr)
        image_arr.CopyInformation(image)
        if prefix is not None:
            clipped_image_path = os.path.join(save_path, prefix + image_file)
        else:
            clipped_image_path = os.path.join(save_path, image_file)
        sitk.WriteImage(image_arr, clipped_image_path)

def convert_label(image_files, image_folder_path, save_path, orginal_label = [1,2,3,4,5], target_label = [1,1,1,1,1]):
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image = sitk.ReadImage(image_path)
        image_arr = sitk.GetArrayFromImage(image)
        
        # assert image_arr.max() == orginal_label[-1]
        # for index, value in enumerate(orginal_label):
        #     image_arr[image_arr == value] = target_label[index]
        image_arr[image_arr>0] = 1
        image_arr = sitk.GetImageFromArray(image_arr)
        image_arr.CopyInformation(image)
        clipped_image_path = os.path.join(save_path, image_file)
        sitk.WriteImage(image_arr, clipped_image_path)

def creat_empty_label(image_files, image_folder_path, save_path):
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image = sitk.ReadImage(image_path)
        image_arr = sitk.GetArrayFromImage(image)
        print(image_file, image_arr.shape)

        zero_arr = np.zeros_like(image_arr)
        
        # assert image_arr.max() == orginal_label[-1]
        # for index, value in enumerate(orginal_label):
        #     image_arr[image_arr == value] = target_label[index]
        image_arr[image_arr>0] = 1
        assert zero_arr.sum() == 0
        image_arr = sitk.GetImageFromArray(zero_arr)
        image_arr.CopyInformation(image)
        clipped_image_path = os.path.join(save_path, image_file)
        sitk.WriteImage(image_arr, clipped_image_path)


if __name__ == '__main__':
    base_in_dir = "/home/data3/whm/dataset/flare25/FLARE-MedFM/processed/1adj_spacing/"
    base_out_dir = "/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/"

    # AllDataTrack_mr_dir=base_in_dir+'train_MRI_unlabeled/LLD-MMRI-3984/'
    # ref_images = os.listdir(AllDataTrack_mr_dir)
    # # mada: +Delay +V +A Phase -pre _DWI T2WI
    # modas = ['amos','OutPhase', 'C+Delay', 'C-pre', 'C+V', 'C+A', 'InPhase', 'DWI', 'T2WI']
    # for moda in modas:
    #     for image in ref_images:
    #         if moda in image:
    #             out_npy_path = base_out_dir+'AllDataTrack/mri/train/'+moda
    #             if not os.path.exists(out_npy_path):
    #                 os.makedirs(out_npy_path)
    #             # to_npy([image], AllDataTrack_mr_dir, out_npy_path, isMRI = True)
    #             if len(os.listdir(out_npy_path)) < 80:
    #                 to_npy([image], AllDataTrack_mr_dir, out_npy_path, isMRI = True)


    # AllDataTrack_mr_dir=base_in_dir+'train_MRI_unlabeled/AMOS-833/'
    # ref_images = os.listdir(AllDataTrack_mr_dir)
    # # mada: +Delay +V +A Phase -pre _DWI T2WI
    # modas = ['amos','OutPhase', 'C+Delay', 'C-pre', 'C+V', 'C+A', 'InPhase', 'DWI', 'T2WI']
    # for moda in modas:
    #     for image in ref_images:
    #         if moda in image:
    #             out_npy_path = base_out_dir+'AllDataTrack/mri/train/'+moda
    #             if not os.path.exists(out_npy_path):
    #                 os.makedirs(out_npy_path)
    #             # to_npy([image], AllDataTrack_mr_dir, out_npy_path, isMRI=True)
    #             if len(os.listdir(out_npy_path)) < 80:
    #                 to_npy([image], AllDataTrack_mr_dir, out_npy_path, isMRI = True)

    AllDataTrack_pet_dir=base_in_dir+'train_PET_unlabeled/'
    ref_images = os.listdir(AllDataTrack_pet_dir)
    # mada:fdg, psma
    modas = ['fdg','psma']
    # modas = ['psma']
    for moda in modas:
        for image in ref_images:
            if moda in image:
                out_npy_path = base_out_dir+'AllDataTrack/pet/train/'+moda
                if not os.path.exists(out_npy_path):
                    os.makedirs(out_npy_path)
                # to_npy([image], AllDataTrack_pet_dir, out_npy_path, isMRI=True)
                if len(os.listdir(out_npy_path)) < 80:
                    to_npy([image], AllDataTrack_pet_dir, out_npy_path, isMRI = True)




    # base_in_dir = "/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/"
    # base_out_dir = "/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/"
    # ct_img_path_50 = base_in_dir+'train_CT_gt_label/imagesTr/'
    # ct_lab_path_50 = base_in_dir+'train_CT_gt_label/labelsTr/'
    # ct_images = os.listdir(ct_img_path_50)
    # to_npy(ct_images, ct_img_path_50, base_out_dir+'AllDataTrack/mri/train/img',isimage = True, isCT = True)
    # ct_labels = os.listdir(ct_lab_path_50)
    # to_npy(ct_labels, ct_lab_path_50, base_out_dir+'AllDataTrack/mri/train/lab',isimage = False, isCT = True)

