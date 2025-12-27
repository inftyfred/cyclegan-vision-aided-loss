from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # Vision-aided loss options for CycleGAN
        # parser.add_argument('--use_vision_aided_loss', action='store_true', help='whether to use vision-aided discriminator')
        # parser.add_argument('--cv_type', type=str, default='clip', help='pretrained model type for vision-aided loss (e.g., clip, dino, swin, vgg)')
        # parser.add_argument('--cv_output_type', type=str, default='conv_multi_level', help='output type for vision-aided discriminator (e.g., conv, conv_multi_level)')
        # parser.add_argument('--cv_loss', type=str, default='multilevel_sigmoid_s', help='loss type for vision-aided discriminator (e.g., sigmoid, multilevel_sigmoid_s, hinge)')
        # parser.add_argument('--cv_lambda', type=float, default=1.0, help='weight for vision-aided loss')
        # parser.add_argument('--cv_diffaug', action='store_true', default=True, help='whether to use DiffAugment in vision-aided discriminator (default: True)')
        # parser.add_argument('--cv_lr', type=float, default=0.0002, help='learning rate for vision-aided discriminator decoder')
        # parser.add_argument('--cv_warmup_iter', type=int, default=0, help='number of warmup iterations before applying vision-aided loss (default: 0)')

        self.isTrain = True
        return parser
