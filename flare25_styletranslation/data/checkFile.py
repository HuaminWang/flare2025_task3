import os
import glob
import SimpleITK as sitk
from tqdm import tqdm


def check_and_clean_nii_gz(directory):
    """
    检查目录下所有.nii.gz文件的完整性，删除损坏文件

    参数:
        directory (str): 要检查的目录路径

    返回:
        list: 被删除的损坏文件绝对路径列表
    """
    # 获取所有.nii.gz文件的绝对路径
    nii_files = glob.glob(os.path.join(directory, '**', '*.nii.gz'), recursive=True)

    deleted_files = []

    # 使用tqdm创建进度条
    for file_path in tqdm(nii_files, desc="检查NIfTI文件"):
        try:
            # 尝试读取文件（仅元数据，不加载像素数据）
            reader = sitk.ImageFileReader()
            reader.SetFileName(file_path)
            reader.ReadImageInformation()  # 只读取头信息，效率更高

            # 可选：完整读取文件（更彻底但更慢）
            # img = sitk.ReadImage(file_path)

        except Exception as e:
            # 捕获任何异常（文件损坏或无法读取）
            try:
                os.remove(file_path)
                abs_path = os.path.abspath(file_path)
                deleted_files.append(abs_path)
                tqdm.write(f"已删除损坏文件: {abs_path} | 错误: {str(e)}")
            except Exception as remove_error:
                tqdm.write(f"删除失败: {file_path} | 错误: {str(remove_error)}")

    return deleted_files

if __name__ == '__main__':
    dataset_dir=  "/home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption"
    deleted_files = check_and_clean_nii_gz(dataset_dir)
    print(f"已删除 {len(deleted_files)} 个损坏的NIfTI文件。")
    print("被删除的文件列表:", deleted_files)

    '''
    检查NIfTI文件:   9%|▊         | 729/8437 [00:02<00:22, 348.20it/s]WARNING: In /tmp/SimpleITK-build/ITK/Modules/IO/NIFTI/src/itkNiftiImageIO.cxx, line 2008
    NiftiImageIO (0x1e5c6090): /home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/train_MRI_unlabeled/AMOS-833/amos_7524_0000.nii.gz has unexpected scales in sform
    检查NIfTI文件:  14%|█▍        | 1220/8437 [00:03<00:21, 332.55it/s]WARNING: In /tmp/SimpleITK-build/ITK/Modules/IO/NIFTI/src/itkNiftiImageIO.cxx, line 2008
    NiftiImageIO (0x1e649080): /home/data3/whm/dataset/flare25/FLARE-MedFM/FLARE-Task3-DomainAdaption/train_MRI_unlabeled/AMOS-833/amos_7713_0000.nii.gz has unexpected scales in sform
    '''