import numpy as np
import random

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t**(n-i)) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def minmax01(sample):
    v0 = sample.min() 
    v1 = sample.max() 

    sample[sample < v0] = v0
    sample[sample > v1] = v1
    sample = (sample - v0) / (v1 - v0) #* 2.0 - 1.0
    return sample
def minmax_11(sample):
    sample = sample * 2.0 - 1.0
    return sample
def nonlinear_transformation(x,minn,maxx, prob=0.5, mul=None):
    x = (x - minn) / (maxx - minn)

    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)

    if random.random() <= 0.5:
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)

    nonlinear_x = np.interp(x, xvals, yvals)

    if mul is not None:
        nonlinear_x = nonlinear_x * mul
    nonlinear_x = minn + nonlinear_x * (maxx - minn)
    return nonlinear_x





def cutmix(data_all, seg_all, p=0.5, alpha=1.0, foreground_labels=[1,2,3,4,5]):
    """
    实现 CutMix 操作，在任意 batch size 的图像中随机交换来自前景的 patch。

    参数:
    - data_all: 形状为 (batch_size, channels, depth, height, width) 的图像数据
    - seg_all: 形状为 (batch_size, channels, depth, height, width) 的标签数据
    - p: 执行 CutMix 的概率
    - alpha: 控制 beta 分布的参数，决定随机裁剪大小
    - foreground_labels: 前景标签的值列表（默认为 [0,1,2,3,4,5]）

    返回:
    - data_all_cutmix: 进行 CutMix 后的图像数据
    - seg_all_cutmix: 进行 CutMix 后的标签数据
    """
    if np.random.rand() > p:
        return data_all, seg_all

    batch_size = data_all.shape[0]
    idx = np.random.permutation(batch_size)
    data_shuffled = data_all[idx]
    seg_shuffled = seg_all[idx]

    # 生成用于 CutMix 的随机 lambda 值
    lam = np.random.beta(alpha, alpha)

    # 获取图像的维度
    _, _, D, H, W = data_all.shape
    cut_rat = np.sqrt(1. - lam)  # 裁剪比例
    cut_D = max(1, int(D * cut_rat))
    cut_H = max(1, int(H * cut_rat))
    cut_W = max(1, int(W * cut_rat))

    data_all_cutmix = data_all.copy()
    seg_all_cutmix = seg_all.copy()

    for i in range(batch_size):
        # 获取当前样本的前景坐标
        # 前景标签包含所有指定的标签值
        foreground_mask = np.isin(seg_all[i, 0], foreground_labels)
        foreground_indices = np.argwhere(foreground_mask)

        if len(foreground_indices) == 0:
            # 如果没有前景，随机选择中心点
            cz = np.random.randint(D)
            cy = np.random.randint(H)
            cx = np.random.randint(W)
        else:
            # 从前景坐标中随机选择一个作为中心点
            cz, cy, cx = foreground_indices[np.random.choice(len(foreground_indices))]

        # 计算裁剪框的边界，确保切片不为空
        bbx1 = np.clip(cx - cut_W // 2, 0, W - 1)
        bbx2 = np.clip(cx + cut_W // 2, bbx1 + 1, W)
        bby1 = np.clip(cy - cut_H // 2, 0, H - 1)
        bby2 = np.clip(cy + cut_H // 2, bby1 + 1, H)
        bbz1 = np.clip(cz - cut_D // 2, 0, D - 1)
        bbz2 = np.clip(cz + cut_D // 2, bbz1 + 1, D)

        # 交换 patch
        data_all_cutmix[i, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2] = data_shuffled[i, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2]
        seg_all_cutmix[i, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2] = seg_shuffled[i, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2]

    return data_all_cutmix, seg_all_cutmix



def copy_paste(data_all, seg_all, p=0.5, paste_classes=[1,2,3,4,5], min_crop_size=16, max_crop_size=64):
    """
    实现 Copy-Paste 操作，从指定类别的前景区域中随机裁剪一个 patch，并粘贴到另一张图像上。

    参数:
    - data_all: 图像数据，形状为 (batch_size, channels, depth, height, width)
    - seg_all: 标签数据，形状为 (batch_size, channels, depth, height, width)
    - p: 执行 Copy-Paste 的概率
    - paste_classes: 需要复制粘贴的类别标签列表
    - min_crop_size: 裁剪区域的最小尺寸（像素）
    - max_crop_size: 裁剪区域的最大尺寸（像素）

    返回:
    - data_all_cp: 经过 Copy-Paste 操作后的图像数据
    - seg_all_cp: 经过 Copy-Paste 操作后的标签数据
    """
    if np.random.rand() > p:
        return data_all, seg_all

    batch_size = data_all.shape[0]
    idx = np.random.permutation(batch_size)
    data_shuffled = data_all[idx]
    seg_shuffled = seg_all[idx]

    data_all_cp = data_all.copy()
    seg_all_cp = seg_all.copy()

    for i in range(batch_size):
        # 获取当前样本中指定类别的前景掩码
        paste_mask = np.isin(seg_all[i, 0], paste_classes)

        if not np.any(paste_mask):
            # 如果当前样本中没有指定的前景类别，跳过
            continue

        # 获取前景体素的坐标
        foreground_indices = np.argwhere(paste_mask)
        z_coords, y_coords, x_coords = foreground_indices[:, 0], foreground_indices[:, 1], foreground_indices[:, 2]

        # 获取前景区域的尺寸范围
        z_range = z_coords.ptp() + 1  # ptp() 返回最大值与最小值之差
        y_range = y_coords.ptp() + 1
        x_range = x_coords.ptp() + 1

        # 计算每个维度的最小裁剪尺寸，不能超过前景区域的尺寸
        min_crop_size_z = min(min_crop_size, z_range)
        min_crop_size_y = min(min_crop_size, y_range)
        min_crop_size_x = min(min_crop_size, x_range)

        # 计算每个维度的最大裁剪尺寸，不能超过前景区域的尺寸
        max_crop_size_z = min(max_crop_size, z_range)
        max_crop_size_y = min(max_crop_size, y_range)
        max_crop_size_x = min(max_crop_size, x_range)

        # 检查是否可以进行裁剪，如果最小尺寸大于最大尺寸，则跳过该样本
        if min_crop_size_z > max_crop_size_z or min_crop_size_y > max_crop_size_y or min_crop_size_x > max_crop_size_x:
            continue

        # 确保 low < high，调用 np.random.randint
        crop_size_z = np.random.randint(min_crop_size_z, max_crop_size_z + 1)
        crop_size_y = np.random.randint(min_crop_size_y, max_crop_size_y + 1)
        crop_size_x = np.random.randint(min_crop_size_x, max_crop_size_x + 1)

        # 从前景坐标中随机选择一个中心点
        center_idx = np.random.choice(len(foreground_indices))
        cz, cy, cx = foreground_indices[center_idx]

        # 计算裁剪区域的边界，确保不超出图像边界
        z_min = max(cz - crop_size_z // 2, 0)
        z_max = min(z_min + crop_size_z, data_all.shape[2])
        y_min = max(cy - crop_size_y // 2, 0)
        y_max = min(y_min + crop_size_y, data_all.shape[3])
        x_min = max(cx - crop_size_x // 2, 0)
        x_max = min(x_min + crop_size_x, data_all.shape[4])

        # 更新实际裁剪尺寸（可能因为边界限制而变化）
        crop_size_z = z_max - z_min
        crop_size_y = y_max - y_min
        crop_size_x = x_max - x_min

        # 从打乱的批次中选择一个目标样本
        target_data = data_shuffled[i]
        target_seg = seg_shuffled[i]

        # 确保粘贴区域不会超出目标样本的边界
        D, H, W = data_all.shape[2:]
        if D - crop_size_z <= 0 or H - crop_size_y <= 0 or W - crop_size_x <= 0:
            # 如果粘贴区域尺寸大于目标图像尺寸，跳过该样本
            continue

        z_start = np.random.randint(0, D - crop_size_z + 1)
        y_start = np.random.randint(0, H - crop_size_y + 1)
        x_start = np.random.randint(0, W - crop_size_x + 1)

        z_end = z_start + crop_size_z
        y_end = y_start + crop_size_y
        x_end = x_start + crop_size_x

        # 获取裁剪区域的掩码
        paste_mask_crop = paste_mask[z_min:z_max, y_min:y_max, x_min:x_max]

        # 执行复制粘贴操作
        data_all_cp[i, :, z_start:z_end, y_start:y_end, x_start:x_end] = np.where(
            paste_mask_crop[np.newaxis, ...],
            data_all[i, :, z_min:z_max, y_min:y_max, x_min:x_max],
            data_all_cp[i, :, z_start:z_end, y_start:y_end, x_start:x_end]
        )

        seg_all_cp[i, :, z_start:z_end, y_start:y_end, x_start:x_end] = np.where(
            paste_mask_crop[np.newaxis, ...],
            seg_all[i, :, z_min:z_max, y_min:y_max, x_min:x_max],
            seg_all_cp[i, :, z_start:z_end, y_start:y_end, x_start:x_end]
        )

    return data_all_cp, seg_all_cp
