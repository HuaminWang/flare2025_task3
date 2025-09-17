import SimpleITK as sitk
import numpy as np
import os
import random

def normalize_image(image):
    """
    Normalize the image to the range [-1, 1].
    """
    image_array = sitk.GetArrayFromImage(image)
    # print(image_array.min(), image_array.max())

    lower_percentile = np.percentile(image_array, 1)
    upper_percentile = np.percentile(image_array, 99)
    image_array = np.clip(image_array, lower_percentile, upper_percentile)

    min_val = np.min(image_array)
    max_val = np.max(image_array)
    normalized_array = 2 * (image_array - min_val) / (max_val - min_val) - 1
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image

def find_body_bbx(normalized_image, threshold=-0.9):
    """
    Find the bounding box of the body part based on a threshold.
    """
    image_array = sitk.GetArrayFromImage(normalized_image)

    mask = image_array > threshold
    non_zero_coords = np.argwhere(mask)

    min_coords = np.min(non_zero_coords, axis=0)
    max_coords = np.max(non_zero_coords, axis=0) + 1  # Add 1 to include the end point

    return min_coords, max_coords

def crop_image(image, min_coords, max_coords):
    """
    Crop the image based on the provided coordinates.
    """
    image_array = sitk.GetArrayFromImage(image)
    cropped_array = image_array[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    cropped_image = sitk.GetImageFromArray(cropped_array)
    original_origin = image.GetOrigin()
    original_spacing = image.GetSpacing()
    new_origin = [original_origin[i] + min_coords[i] * original_spacing[i] for i in range(3)]
    cropped_image.SetOrigin(new_origin)
    cropped_image.SetSpacing(original_spacing)
    cropped_image.SetDirection(image.GetDirection())
    return cropped_image

def resample_image(image, new_spacing, reference_image=None, is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    if reference_image is not None:
        new_size = reference_image.GetSize()
    else:
        new_size = [
            int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
            int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
            int(round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
        ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    # resample.SetDefaultPixelValue(image.GetPixelIDValue())


    if is_label:
        resample.SetDefaultPixelValue(0)  # 标签背景值为0
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        image_array = sitk.GetArrayViewFromImage(image)  # 高效获取数组
        min_val = image_array.min()  # 计算最小值作为背景
        resample.SetDefaultPixelValue(float(min_val))
        resample.SetInterpolator(sitk.sitkLinear)
    
    new_image = resample.Execute(image)
    return new_image

def center_pad_crop(image, target_size=512, is_label=False):
    """
    Center crop or pad the image to the target size (only for height and width).
    For depth (z-axis), we keep the original size.
    For images, pad with the minimum value of the original image.
    For labels, pad with 0.
    """
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)  # shape: (D, H, W)
    D, H, W = array.shape

    # Determine padding or cropping needed for height and width
    pad_h = max(target_size - H, 0)
    pad_w = max(target_size - W, 0)
    crop_h = max(H - target_size, 0)
    crop_w = max(W - target_size, 0)

    # Calculate crop or pad parameters
    start_h = crop_h // 2
    start_w = crop_w // 2
    end_h = start_h + min(H, target_size)
    end_w = start_w + min(W, target_size)

    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left

    # Crop first if needed
    cropped_array = array[:, start_h:end_h, start_w:end_w]
    ch, cw = cropped_array.shape[1:]

    # Then pad if needed
    if pad_h > 0 or pad_w > 0:
        if is_label:
            fill_value = 0
        else:
            fill_value = np.min(array)

        new_array = np.full((D, target_size, target_size), fill_value, dtype=array.dtype)
        insert_h = slice(pad_h_top, pad_h_top + ch)
        insert_w = slice(pad_w_left, pad_w_left + cw)
        new_array[:, insert_h, insert_w] = cropped_array
    else:
        new_array = cropped_array

    # Create new image
    new_image = sitk.GetImageFromArray(new_array)

    # Update origin to account for cropping/padding
    original_origin = list(image.GetOrigin())
    original_spacing = image.GetSpacing()
    original_origin[0] += start_w * original_spacing[0] - pad_w_left * original_spacing[0]
    original_origin[1] += start_h * original_spacing[1] - pad_h_top * original_spacing[1]

    new_image.SetOrigin(original_origin)
    new_image.SetSpacing(original_spacing)
    new_image.SetDirection(image.GetDirection())

    return new_image


def reorient_to_lpi(image):
    # Reorient the image
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("LPI")
    reoriented_image = orient_filter.Execute(image)

    return reoriented_image

def process_images(input_image_folder, input_label_folder, output_image_folder, output_label_folder, new_spacing):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    imagesss = os.listdir(input_image_folder)

    is_lab_exist=input_label_folder is not None and os.path.exists(input_label_folder)
    if is_lab_exist:
        if not os.path.exists(output_label_folder):
            os.makedirs(output_label_folder)

    for image_filename in sorted(imagesss):
        if image_filename.endswith('.nii.gz'):
            image_path = os.path.join(input_image_folder, image_filename)
            if is_lab_exist:
                label_filename = image_filename.replace('_0000', '')
                label_path = os.path.join(input_label_folder, label_filename)
            output_image_path = os.path.join(output_image_folder, image_filename)
            if is_lab_exist:
                output_label_path = os.path.join(output_label_folder, label_filename)
            print(image_path,"-->", output_image_path)
            # print(output_image_path,'6666')
            # try:
            # if not os.path.exists(output_image_path):
            # if 'Phase' in output_image_path or 'DWI' in output_image_path or 'T2WI' in output_image_path:
            #     # if not os.path.exists(output_image_path):
            #     if True:


            # Read the image and label
            image = sitk.ReadImage(image_path)
            if is_lab_exist:
                label = sitk.ReadImage(label_path)

            # Reorient to LPI at the beginning
            image = reorient_to_lpi(image)
            if is_lab_exist:
                label = reorient_to_lpi(label)

            # Normalize the image
            normalized_image = normalize_image(image)

            # Find the body bounding box
            min_coords, max_coords = find_body_bbx(normalized_image)
            # print("find_body_bbx:", min_coords, max_coords)

            # Crop the image and label
            cropped_image = crop_image(image, min_coords, max_coords)
            if is_lab_exist:
                cropped_label = crop_image(label, min_coords, max_coords)

            # Resample the cropped image and label to the new spacing
            resampled_cropped_image = resample_image(cropped_image, new_spacing, is_label=False)

            if is_lab_exist:
                resampled_cropped_label = resample_image(cropped_label, new_spacing, reference_image=resampled_cropped_image, is_label=True)


            # # todo: center crop and pad to fixed size, i.e., N*512*512
            # center_crop_size = 512
            # center_pad_crop(resampled_cropped_image, is_label=False)
            # center_pad_crop(resampled_cropped_label, is_label=True)
            # resampled_cropped_image = center_pad_crop(resampled_cropped_image, target_size=512, is_label=False)
            # Save the resampled image and label
            sitk.WriteImage(resampled_cropped_image, output_image_path)
            if is_lab_exist:
                # resampled_cropped_label = center_pad_crop(resampled_cropped_label, target_size=512, is_label=True)
                resampled_cropped_label.CopyInformation(resampled_cropped_image)
                sitk.WriteImage(resampled_cropped_label, output_label_path)
            # print(f"Processed and saved: {output_image_path} and {output_label_path}")



            # except Exception as e:
            #     print('error')
            #     # print(f"Failed to process {image_filename} )
def main():
    # Example usage
    # input_image_folder = "/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/train_CT_gt_label/imagesTr/"
    # input_image_folder = "/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/coreset_train_unlabeled_MRI_PET/MRI_unlabeled_100_random/"
    # input_label_folder = "/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/train_CT_gt_label/labelsTr/"
    #
    # out_base="/home/data3/whm/dataset/flare25/FLARE-MedFM/processed/1adj_spacing/"
    # output_image_folder = out_base+ "train_CT_gt_label/imagesTr_cpd/"
    # output_label_folder = out_base+ "train_CT_gt_label/labelsTr_cpd/"
    input_image_folder = "/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/coreset_train_unlabeled_MRI_PET/PET_unlabeled_100_random/"
    input_label_folder = None

    out_base="/home/data3/whm/dataset/flare25/FLARE-MedFM/processed/2adj_spacing_test/"
    output_image_folder = out_base+ "coreset_train_unlabeled_MRI_PET_lpi/PET_unlabeled_100_random/"
    output_label_folder = None #out_base+ "train_CT_gt_label/labelsTr_cpd/"
    new_spacing = [0.78125, 0.78125, 2.5]  # Example new spacing
    process_images(input_image_folder, input_label_folder, output_image_folder, output_label_folder, new_spacing)

def main_all():
    import threading
    from os.path import join as ospj
    in_base_dir = r'/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption'
    out_base_dir = r'/home/data3/whm/dataset/flare25/FLARE-MedFM/processed/1adj_spacing'
    new_spacing = [0.78125, 0.78125, 2.5]

    target_saving_paths = [

                            # (
                            #     ospj(in_base_dir, 'train_CT_gt_label/imagesTr'),
                            #     ospj(in_base_dir, 'train_CT_gt_label/labelsTr'),
                            #     ospj(out_base_dir, 'train_CT_gt_label/imagesTr'),
                            #     ospj(out_base_dir, 'train_CT_gt_label/labelsTr')
                            # ),

                            # (
                            #     ospj(in_base_dir, 'train_CT_pseudolabel/imagesTr'),
                            #     None,
                            #     ospj(out_base_dir, 'train_CT_pseudolabel/imagesTr'),
                            #     None
                            # ),

                            (
                                ospj(in_base_dir, 'train_MRI_unlabeled/AMOS-833'),
                                None,
                                ospj(out_base_dir, 'train_MRI_unlabeled/AMOS-833'),
                                None
                            ),

                            (
                                ospj(in_base_dir, 'train_MRI_unlabeled/LLD-MMRI-3984'),
                                None,
                                ospj(out_base_dir, 'train_MRI_unlabeled/LLD-MMRI-3984'),
                                None
                            ),

                            (
                                ospj(in_base_dir, 'train_PET_unlabeled'),
                                None,
                                ospj(out_base_dir, 'train_PET_unlabeled'),
                                None
                            ),

             ]


    def worker(input_image_folder, input_label_folder, output_image_folder, output_label_folder):
        process_images(input_image_folder, input_label_folder, output_image_folder, output_label_folder, new_spacing)

    threads = []
    for (in_img_dir, in_lab_dir, out_img_dir, out_lab_dir) in target_saving_paths:
        t = threading.Thread(target=worker, args=(in_img_dir, in_lab_dir, out_img_dir, out_lab_dir))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

if __name__ == '__main__':
    # main()
    main_all()