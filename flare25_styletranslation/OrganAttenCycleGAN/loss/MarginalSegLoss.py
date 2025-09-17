import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

softmax_helper = lambda x: F.softmax(x, 1)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class MarginalSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, class_weight=None, batch_dice=False, smooth=1.):
        """
        """
        super(MarginalSoftDiceLoss, self).__init__()
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.class_weight = class_weight

    def forward(self, x, y, ignore_label=None, loss_mask=None):
        shp_x = x.shape
        if self.class_weight is not None and self.class_weight.device != x.device:
            self.class_weight = self.class_weight.to(x.device.index)

        if self.batch_dice:
            print('partial label is available, batch_dice cannot be used, change it to False')
            self.batch_dice = False
        axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        dice_class = []
        indictor = torch.cat([torch.ones((1,)), torch.zeros((x.size(1)-1,))]).to(x.device.index)
        for i in range(x.size(0)):
            this_x = x[i]
            this_y = y[i]
            this_partial_label = ignore_label[i]
            if loss_mask is not None:
                this_mask = loss_mask[i]
                if len(this_mask.size()) != len(this_x.size()):
                    this_mask = this_mask.unsqueeze(0)
            else:
                this_mask = None

            new_x = torch.zeros_like(this_x).to(this_x.device.index)
            if all(this_partial_label == 1):
                new_x = this_x
            else:
                new_x[0] = this_x[this_partial_label == 0].sum(0)
                new_x[1:] = this_x[1:]

            if loss_mask is not None:
                this_mask = this_mask.unsqueeze(0)

            tp, fp, fn, _ = get_tp_fp_fn_tn(new_x.unsqueeze(0), this_y.unsqueeze(0), axes, this_mask, False)
            nominator = 2 * tp + self.smooth
            denominator = 2 * tp + fp + fn + self.smooth
            dc = nominator / denominator

            dc = dc.squeeze(0) * this_partial_label + dc.squeeze(0) * (1-this_partial_label) * indictor
            if self.class_weight is not None:
                dc = self.class_weight * dc
            dice_class.append(dc)

        dice_class = torch.stack(dice_class)
        with torch.no_grad():
            tmp = ignore_label.clone()
            tmp[:, 0] = 1
            class_count = torch.sum(tmp, 0)
        dice_class = torch.sum(dice_class, 0)[class_count != 0] / class_count[class_count != 0]

        return -1.0 * dice_class.mean()

class Marginal_DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, class_weight=None, aggregate="sum", weight_ce=1, weight_dice=1, log_dice=False):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Marginal_DC_and_CE_loss, self).__init__()
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate

        if class_weight is not None:
            class_weight = torch.tensor(class_weight).float()
            self.nll = nn.NLLLoss(reduction='none', weight=class_weight.cuda())
        else:
            self.nll = nn.NLLLoss(reduction='none')

        self.dc = MarginalSoftDiceLoss(apply_nonlin=softmax_helper, class_weight=class_weight, **soft_dice_kwargs)

    def forward(self, net_output, target, ignore_label=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if ignore_label is None:
            ignore_label = torch.ones(net_output.size()[:2])
            ignore_label = ignore_label.to(net_output.device.index)

        dc_loss = self.dc(net_output, target, ignore_label=ignore_label)
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        net_output = softmax_helper(net_output)

        if len(net_output.size()) == len(target.size()):
            if all(i == j for (i, j) in zip(net_output.size(), target.size())):
                _, target = torch.max(target, dim=1)
            else:
                target = target.squeeze(1)

        B, C, H, W = net_output.size()
        ce_losses = []
        for i in range(B):
            this_output = net_output[i]
            this_target = target[i]
            this_partial_label = ignore_label[i]

            new_output = torch.zeros_like(this_output).to(this_output.device.index)
            if all(this_partial_label == 1):
                new_output = this_output
            else:
                new_output[0] = this_output[this_partial_label == 0].sum(0)
                new_output[1:] = this_output[1:]

            new_output = torch.log(new_output + 1e-8)
            ce_loss = self.nll(new_output.unsqueeze(0), this_target.unsqueeze(0).long())
            ce_loss = torch.mean(ce_loss, (1, 2))

            ce_losses.append(ce_loss)
        ce_loss = torch.cat(ce_losses).mean()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

if __name__ == '__main__':
    import numpy as np
    x = torch.from_numpy(np.random.random(size=[2, 4, 10, 10])).float()
    y = torch.from_numpy(np.random.randint(0, 4, size=(2, 10, 10)))
    loss = Marginal_DC_and_CE_loss(soft_dice_kwargs={})

    l = loss(x, y)
    print(l)