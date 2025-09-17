import os
import numpy as np
import SimpleITK as sitk
import argparse

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
            # print(f'raw label: {label} --> target label: {label_mapping[label]}')
            modified_seg[seg == label] = label_mapping[label]

    return modified_seg

def walk_dir_affine(raw_dir, save_dir, Affinefile=None):

    labMapping = getMapping(Affinefile) if Affinefile else None

    """processing all nii.gz in raw_dir"""
    os.makedirs(save_dir, exist_ok=True)
    for root, _, files in os.walk(raw_dir):
        for file in files:
            lowercase_file = file.lower()
            #if 'fdg' not in lowercase_file and 'psma' not in lowercase_file:
            if not any(keyword in lowercase_file for keyword in ['fdg', 'psma', 'pet']):
                continue
            if file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                arr, original_img = readnii(file_path)

                try:

                    modified_seg = AffineForeground(arr, labMapping)

                    new_img = sitk.GetImageFromArray(modified_seg)
                    new_img.CopyInformation(original_img)

                    save_path = os.path.join(save_dir, file.replace('_mask', ''))  # remove postfix "_mask"
                    sitk.WriteImage(new_img, save_path)
                    # print(f"Processed: {file}")

                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Apply affine transformation to prediction labels.")
    parser.add_argument('-raw_dir', type=str, default='./outputs/',
                        help='Path to the input directory containing raw prediction labels.')
    parser.add_argument('-save_dir', type=str, default='./outputs/',
                        help='Path to the output directory where transformed labels will be saved.')
    parser.add_argument('-affine_file', type=str, default='./flare25_pet_lab_affine.csv',
                        help='CSV file containing affine transformation parameters.')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    walk_dir_affine(args.raw_dir, args.save_dir, args.affine_file)