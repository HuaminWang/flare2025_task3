# -*- coding: utf-8 -*-
"""
Created: Nov 2018

@author: Jue Jiang

Date modified: July 7 2020
@author: Harini Veeraraghavan
Description of changes: Cleaning up code and documentation

"""

# -*- coding: utf-8 -*-

import torch.nn.functional as F
import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from util import util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from loss.MarginalSegLoss import Marginal_DC_and_CE_loss
from loss.WGANLoss import calc_gradient_penalty


class Organ_attention_only_cycleGAN(BaseModel):
    def name(self):
        return 'Organ_attention'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        if opt.isTrain:
            self.input_A = self.Tensor(nb, opt.input_nc, size, size)  # input A
            self.input_B = self.Tensor(nb, opt.output_nc, size, size)  # input B
            self.input_A = self.input_A.cuda()
            self.input_B = self.input_B.cuda()

        else:
            self.test_A = self.Tensor(nb, opt.output_nc, size, size)  # input B
            self.test_B = self.Tensor(nb, opt.output_nc, size, size)  # input B

            self.test_A = self.test_A.cuda()
            self.test_B = self.test_B.cuda()

        self.netG_A = networks.define_G(1, 1,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)
        self.netG_B = networks.define_G(1, 1,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.isTrain:
            use_sigmoid = True
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            # use_sigmoid = opt.no_lsgan
            if opt.gan_type == 'lsgan' or opt.gan_type == 'wgan-gp':
                use_sigmoid = False
            else:
                use_sigmoid = True

            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            if self.isTrain:
                print('loading pretrained G_A......')
                self.load_network(self.netG_A, 'G_A', which_epoch)
                for name, param in self.netG_A.named_parameters():
                    print(param)
                    break
                print('loading pretrained D_A.......')
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if not opt.only_resume_GDA:
                    self.load_network(self.netG_B, 'G_B', which_epoch)
                    self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers

            if opt.frozen_GDA:
                self.optimizer_G = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), amsgrad=True)
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999), amsgrad=True)
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr * opt.D_lr_mult,
                                                      betas=(opt.beta1, 0.999), amsgrad=True)

            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr * opt.D_lr_mult,
                                                  betas=(opt.beta1, 0.999), amsgrad=True)

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if not opt.frozen_GDA:
                self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)

        print('-----------------------------------------------')

    def net_G_A_load_weight(self, weight):
        self.load_network(self.netG_A, 'G_A', weight)

    def net_D_A_load_weight(self, weight):
        self.load_network(self.netD_A, 'D_A', weight)

    def net_D_B_load_weight(self, weight):
        self.load_network(self.netD_B, 'D_B', weight)

    def get_curr_lr(self):
        self.cur_lr = self.optimizer_Seg_A.param_groups[0]['lr']

    def set_test_input(self, input):
        if self.opt.G_A:
            input_A11 = input['CT']
            self.test_A.resize_(input_A11.size()).copy_(input_A11)
        if self.opt.Seg:
            input_B11 = input['MR']
            self.test_B.resize_(input_B11.size()).copy_(input_B11)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'

        input_A1 = input['CT']

        input_A11, input_A12 = torch.split(input_A1, input_A1.size(1) // 2, dim=1)  # mt_BN use
        input_A12 = input_A12.long()

        input_B1 = input['MRI']

        if input_B1.size(1) > 1:
            input_B11, input_B12 = torch.split(input_B1, input_B1.size(1) // 2, dim=1)  # mt_BN use
            input_B12 = input_B12.long()
        else:
            input_B11, input_B12 = input_B1, None

        self.input_A.resize_(input_A11.size()).copy_(input_A11)
        self.input_B.resize_(input_B11.size()).copy_(input_B11)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test_forward(self):
        return_results = {}
        if self.opt.G_A:
            real_A = Variable(self.test_A)
            fake_B = self.netG_A(real_A)
            return_results['fake_B'] = fake_B
        return return_results

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        # loss_D.backward()
        return loss_D

    def backward_D_basic_A(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # backward
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)  # same to self.fake_B_local_pool

        loss_D_A = self.backward_D_basic_A(self.netD_A, self.real_B, fake_B)
        Total_loss_DA = loss_D_A
        if not self.opt.frozen_GDA:
            if self.opt.gan_type == 'wgan-gp':
                use_gpu = True if fake_B.device.type == 'cuda' else False
                gradient_penalty = calc_gradient_penalty(self.netD_A, self.real_B.data, fake_B.data, use_cuda=use_gpu)
                Total_loss_DA += self.opt.lambda_gp * gradient_penalty
            Total_loss_DA.backward()
        self.loss_D_A = loss_D_A.item()  # .data[0]
        self.fake_B_in_D = fake_B.data[0]
        return Total_loss_DA


    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        Total_loss_DB = loss_D_B
        if self.opt.gan_type == 'wgan-gp':
            use_gpu = True if fake_A.device.type == 'cuda' else False
            gradient_penalty = calc_gradient_penalty(self.netD_B, self.real_A.data, fake_A.data, use_cuda=use_gpu)
            Total_loss_DB += self.opt.lambda_gp * gradient_penalty
            # print('D_B Discrimination:{:.4f}, gradient penalty:{:.4f}'.format(loss_D_B, gradient_penalty))
        Total_loss_DB.backward()
        self.loss_D_B = loss_D_B.item()  # .data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.item()  # .data[0]
            self.loss_idt_B = loss_idt_B.item()  # .data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        if self.opt.gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), max_norm=self.opt.max_norm)
            torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), max_norm=self.opt.max_norm)

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.item()  # .data[0]
        self.loss_G_B = loss_G_B.item()  # .data[0]
        self.loss_cycle_A = loss_cycle_A.item()  # .data[0]
        self.loss_cycle_B = loss_cycle_B.item()  # .data[0]

    def optimize_parameters(self, flag):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        for name, param in self.netG_A.named_parameters():
            print(param)
            break

        if flag:
            if not self.opt.frozen_GDA:
                self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            if not self.opt.frozen_GDA:
                self.optimizer_D_A.step()

            self.optimizer_D_B.zero_grad()
            self.backward_D_B()
            self.optimizer_D_B.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B)])

        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

        return ret_visuals

    def save(self, label):
        if not self.opt.only_resume_GDA:
            self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
            self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

