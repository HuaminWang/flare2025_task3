import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import argparse

def lab_voter(esb_dir: list, out_dir, threhold=0.5):
    os.makedirs(out_dir, exist_ok=True)

    # 获取所有预测文件名
    pred_files = sorted(glob(os.path.join(esb_dir[0], '*.nii.gz')))

    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        predictions = []

        for model_dir in esb_dir:
            model_pred_path = os.path.join(model_dir, filename)
            pred_img = sitk.ReadImage(model_pred_path)
            pred_array = sitk.GetArrayFromImage(pred_img)
            predictions.append(pred_array)

        # 转换为numpy数组，形状：(n_models, D, H, W)
        predictions_np = np.stack(predictions, axis=0)

        # 获取所有可能的标签值（包含所有模型中的所有标签）
        unique_labels = np.unique(predictions_np)
        unique_labels = unique_labels[unique_labels != 0]  # 忽略背景0

        # 初始化输出标签图像
        label_vote = np.zeros_like(predictions_np[0], dtype=np.uint8)

        for label in unique_labels:
            # 对每个类别进行投票计数
            label_votes = np.sum(predictions_np == label, axis=0)

            # 创建当前类别满足共识阈值的掩码
            label_mask = label_votes >= (len(esb_dir) * threhold)

            # 只在未标注的位置填充，防止多类冲突
            conflict_mask = (label_mask & (label_vote == 0))
            label_vote[conflict_mask] = label

        # 保存结果图像
        result_img = sitk.GetImageFromArray(label_vote)
        result_img.CopyInformation(pred_img)
        sitk.WriteImage(result_img, os.path.join(out_dir, filename))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply voting fusion to ensemble predictions.")
    parser.add_argument('-esb_dir', nargs='+', required=True, help='List of ensemble prediction directories')
    parser.add_argument('-out_dir', required=True, help='Output directory for fused label maps')
    parser.add_argument('-threhold', type=float, default=0.5, help='Consensus threshold ratio (0~1)')

    args = parser.parse_args()
    lab_voter(esb_dir=args.esb_dir, out_dir=args.out_dir, threhold=args.threhold)

    # Example:
    # python lab_voter.py -esb_dir model1_preds model2_preds model3_preds -out_dir fused_results -threhold 0.5

    # 1. 无标签图像 respacing
    # 2. top2模型预测
    # 3. 评分高 模型相似度样本选择, top500
    # 4. 选中高相似度样本的前景区域过滤， 多个模型前景投票, threhold=0.5
    # 5. 再次训练