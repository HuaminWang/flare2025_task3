import os
import numpy as np
import SimpleITK as sitk

def readnii(file_path):
    """读取nii.gz文件为三维数组，并返回数组及SimpleITK图像对象"""
    img = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(img)  # 数组形状为 (z, y, x)
    return arr, img

def getMapping(Affinefile):
    import pandas as pd
    # Read the Excel file
    df = pd.read_csv(Affinefile)

    # Extract the 'before' --> 'after' columns
    raw_label = df['before'].values.astype(np.uint8)
    target_label = df['after'].values.astype(np.uint8)

    # Create a mapping from algorithm to hospital labels
    label_mapping = dict(zip(raw_label, target_label))
    return label_mapping

def AffineForeground(seg, label_mapping):

    # Get unique labels in the segmentation array
    unique_labels = np.unique(seg)
    unique_labels = unique_labels[unique_labels>0]


    # Create a list for labels that need remapping
    labels_to_remap = []

    # Check if all unique labels need mapping and match algorithm with hospital labels
    all_match = True
    for label in unique_labels:
        if label in label_mapping and label_mapping[label] != label:
            labels_to_remap.append(label)
            all_match = False

    # If all unique labels map correctly, return None
    if all_match:
        return seg

    # Create a modified segmentation array
    modified_seg = np.copy(seg)
    for label in labels_to_remap:
        if label in label_mapping:
            print(f'raw label: {label} --> target label: {label_mapping[label]}')
            modified_seg[seg == label] = label_mapping[label]

    return modified_seg

def walk_dir_affine(raw_dir, save_dir, Affinefile=None):

    labMapping = getMapping(Affinefile) if Affinefile else None

    """processing all nii.gz in raw_dir"""
    os.makedirs(save_dir, exist_ok=True)
    for root, _, files in os.walk(raw_dir):
        # for file in files[:5]:
        for file in sorted(files):
            if file.endswith('.nii.gz'):

                file_path = os.path.join(root, file)
                arr, original_img = readnii(file_path)

                try:

                    modified_seg = AffineForeground(arr, labMapping)

                    new_img = sitk.GetImageFromArray(modified_seg)
                    new_img.CopyInformation(original_img)

                    save_path = os.path.join(save_dir, file.replace('_mask', ''))  # remove postfix "_mask"
                    sitk.WriteImage(new_img, save_path)
                    print(f"Processed: {file}")

                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")


if __name__ == "__main__":
    Affinefile = r'flare25_pet_lab_affine.csv'
    # base_in_dir = '/home/data3/whm/dataset/flare25/FLARE-MedFM/processed/1adj_spacing/'
    # base_out_dir = '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice/temp/'
    base_in_dir = '/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/'
    base_out_dir = '/home/data3/whm/dataset/flare25/FLARE-MedFM/npy_slice_before/temp/'

    raw_dir = base_in_dir+'train_CT_gt_label/labelsTr/'
    save_dir = base_out_dir+'train_CT_gt_label/labelsTr/' # affine 14 organs -> 4 organs for pet lab attention
    os.makedirs(save_dir, exist_ok=True)

    walk_dir_affine(raw_dir, save_dir, Affinefile)