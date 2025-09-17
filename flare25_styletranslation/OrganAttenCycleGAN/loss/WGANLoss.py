import torch
import torch.nn as nn
import torch.autograd as autograd

class WGANLoss(nn.Module):
    def __int__(self):
        super(WGANLoss, self).__init__()

    def forward(self, x, target_is_real):
        if target_is_real:
            return -1.0 * torch.mean(x)
        else:
            return torch.mean(x)

def calc_gradient_penalty(netD, real_data, fake_data, use_cuda=True):
    #print real_data.size()
    alpha_size = [real_data.size(0), ] + [1] * len(real_data.size()[1:])
    alpha = torch.rand(alpha_size)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
