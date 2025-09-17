from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--organ_num', type=int, default=8, help='# of segmented organs.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='30',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--G_A', action='store_false', default=True, help='whether to output the generator results')
        self.parser.add_argument('--Seg', action='store_true', default=False, help='whether to output the segmentation results')
        self.parser.add_argument('--Net', type=str, default='Unet', help='the architecture of the segmentation net')
        self.parser.add_argument('--Organ_Attention_Type', type=str, default='concat', help='Attention type: concat|dot|add' )
        #self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.isTrain = False
