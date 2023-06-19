# -*- coding: utf-8 -*-
# @Time    : 3/6/21 9:58 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# -*- coding: utf-8 -*-

import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args(jupyter=False):
    parser = argparse.ArgumentParser(description="SSc score prediction.")

    # Network
    parser.add_argument('--net', choices=('vgg11_3d', 'vit3', 'vgg16_3d', 'vgg19_3d', 'r3d_resnet', 'cnn3fc1', 'cnn4fc2',
                                          'cnn5fc2', 'cnn6fc2', 'cnn2fc1', 'cnn3fc2', 'r3d_18', 'slow_r50',
                                          'slowfast_r50', 'x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l', 'pointnet_reg','pointnet2_reg',
                                          'vgg11_3d', 'pointnext', 'pointmlp_reg', 'mlp_reg'),  # 'r2plus1d_18' out of memory
                        help='network name', type=str, default='pointnet2_reg') 
    
    # Point cloud network configuration
    parser.add_argument('--cfg', help='fpath of cfg',type=str, default='SSc_vessel/pointnext-s.yaml')  # ori = 40
    parser.add_argument('--fc2_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=1024)
    parser.add_argument('--fc1_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=1024)
    parser.add_argument('--pointnet_fc_ls', help='a parameter list for fully connected layers. First number is the feature number after feature extraction', type=str, default="1024-512-256")
    parser.add_argument('--dp_fc1_flag', help='dropout for fc1',
                        type=boolean_string, default=True)
    parser.add_argument('--npoint_base', help='base of npoint',
                        type=int, default=512)  # ori = 512
    parser.add_argument('--radius_base', help='base of radius',
                        type=int, default=40)  # ori = 40
    parser.add_argument('--nsample_base', help='base of nsample',
                        type=int, default=64)  # ori = 64
    parser.add_argument('--width', help='width',
                        type=int, default=16)  # ori = 64
    parser.add_argument('--radius_scaling', help='base of radius',
                        type=float, default=2)  # ori = 2
    parser.add_argument('--sa_layers', help='sa_layers',
                        type=int, default=2)  # ori = 3

    # data
    # common data
    parser.add_argument('--batch_size', help='batch_size',
                        type=int, default=5)
    parser.add_argument('--ct_sp', help='space', type=str,
                        choices=('ori', '1.0', '1.5'), default='1.5')
    parser.add_argument('--kfold_seed', help='kfold_seed',
                        type=int, default=711)
    parser.add_argument('--test_pat', help='testing patients', choices=(
        'zhiwei77', 'random', 'random_as_ori'), type=str, default='random_as_ori')  # 
    parser.add_argument('--input_mode', help='what to input. This influence the network architecture and dataloader/trans. I do not split it to two parameters. Otherwise, the two need to be synthesized', 
        choices=('ct', 'ct_masked_by_torso', 'ct_left','ct_masked_by_lung','ct_masked_by_left_lung', 'ct_masked_by_right_lung', 
        'ct_right','ct_left_in_lung', 'ct_right_in_lung','ct_upper','ct_lower', 'ct_front', 'ct_back','ct_upper_in_lung',
        'ct_lower_in_lung', 'ct_front_in_lung', 'lung_masks', 'ct_back_in_lung', 'vessel', 'ct_masked_by_vessel',  
        'ct_masked_by_vessel_dilated1', 'ct_masked_by_vessel_dilated2', 'ct_masked_by_vessel_dilated3', 'ct_masked_by_vessel_dilated4',
        'IntrA_cls_pcd', 'modelnet40_pcd', 'lung_mask_pcd', 'vessel_skeleton_pcd'),
        type=str, default='vessel_skeleton_pcd')
    parser.add_argument('--target', help='target prediction', type=str,
                        default='DLCOc_SB-FEV1-FVC-TLC_He')  # FVC-DLCO_SB-FEV1-TLC_He-Age-Height-Weight--DLCOc/pred-FEV1/pred-FVC/predNew-TLC/pred DLCOcPP-FEV1PP-FVCPP-TLCPP
    parser.add_argument(
        '--workers', help='number of workers for dataloader', type=int, default=12)

    # for gird image data
    parser.add_argument('--balanced_sampler', help='balanced_sampler', type=boolean_string, default='True')
    parser.add_argument('--crop_foreground', help='load lung mask, apply RandomCropForegroundd',
                        type=boolean_string, default='True')
    parser.add_argument(
        '--z_size', help='length of patch along z axil ', type=int, default=240)
    parser.add_argument(
        '--y_size', help='length of patch along y axil ', type=int, default=240)
    parser.add_argument(
        '--x_size', help='length of patch along x axil ', type=int, default=240)
    parser.add_argument('--pad_ratio', help='padding ratio',
                        type=float, default=1.5)

    # for point cloud data
    # parser.add_argument('--dataset', help='dataset name', choices=('modelnet40', 'vessel_pcd', 'ct', 'lung_mask_pcd'), type=str, default='lung_mask_pcd')
    parser.add_argument('--set_all_r_to_1', help='set all r values to 1 to avoid the influence of R', type=boolean_string, default='False')
    parser.add_argument('--set_all_xyz_to_1', help='set all xyz values to 1 to avoid the influence of position of points', type=boolean_string, default='False')
    parser.add_argument('--scale_range', help='scale range', type=str, default='0.5-1.5')
    parser.add_argument('--shift_range', help='shift range', type=float, default=0)
    parser.add_argument('--PNB', help='points number for each image', type=int, default=28000)  # maximum nmber: 140 000
    parser.add_argument('--FPS_input', help='Fartest point sample input', type=boolean_string, default='False')
    parser.add_argument('--use_normals_only1', help='use_normals_only1 to be comparible with vessel radius', type=boolean_string, default='True')

    parser.add_argument('--repeated_sample', help='if apply repeated sampling to get PNB points?', type=boolean_string, default='False')
    parser.add_argument('--position_center_norm', help='if use the relative coordinates: center point is 0,0,0', type=boolean_string, default='True')


    # training parameters
    parser.add_argument('--mode', choices=('train', 'infer',
                        'continue_train'), help='mode', type=str, default='train')
    parser.add_argument('--pretrained_id', help='id used for inference, or continue_train',
                        type=str, default="0")  # SSc-852-853-854-855, 1504-1505-1510-1515, 2371-2375-2379-23， 2958-2959-2960-2961 3020-3021-3022-3023
    # parser.add_argument('--reload_jobid', help='jobid used for inference, or continue_train', type=int, default=0)
    parser.add_argument('--pretrained_imgnet', help='if pretrained from imagenet',
                        type=boolean_string, default='False')
    parser.add_argument('--total_folds', choices=(1, 4, 5),
                        help='4-fold training', type=int, default=4)
    parser.add_argument('--fold', choices=(1, 2, 3, 4),
                        help='1 to 4', type=int, default=1)
    parser.add_argument(
        '--valid_period', help='how many epochs between 2 validation', type=int, default=5)
    parser.add_argument('--loss', choices=('mse', 'mae', 'smooth_mae', 'mse+mae', 'msehigher', 'mse_regular'), help='mode', type=str,
                        default='mse')
    parser.add_argument('--mat_diff_loss_scale',
                        help='scale for another loss', type=float, default=0)
    parser.add_argument('--epochs', help='total epochs', type=int, default=500)
    parser.add_argument('--weight_decay', help='L2 regularization', type=float,
                        default=0.001)  # must be a float number !
    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.001)
    parser.add_argument('--adamw', help='adamw optimizer',
                        type=boolean_string, default='False')
    parser.add_argument('--cosine_decay', help='cosine_decay',
                        type=boolean_string, default='False')
    
    # for point-ULIP 
    # parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    # parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    # parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    # parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    # parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    # parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    # parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # # Model
    # parser.add_argument('--model', default='ULIP_PN_SSG', type=str)
    # # Training
    # parser.add_argument('--epochs', default=250, type=int)
    # parser.add_argument('--warmup-epochs', default=1, type=int)
    # parser.add_argument('--start-epoch', default=0, type=int)
    # parser.add_argument('--batch-size', default=64, type=int,
    #                     help='number of samples per-device/per-gpu')
    # parser.add_argument('--lr', default=3e-3, type=float)
    # parser.add_argument('--lr-start', default=1e-6, type=float,
    #                     help='initial warmup lr')
    # parser.add_argument('--lr-end', default=1e-5, type=float,
    #                     help='minimum final lr')
    # parser.add_argument('--update-freq', default=1, type=int,
    #                     help='optimizer update frequency (i.e. gradient accumulation steps)')
    # parser.add_argument('--wd', default=0.1, type=float)
    # parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    # parser.add_argument('--eps', default=1e-8, type=float)
    # parser.add_argument('--eval-freq', default=1, type=int)
    # parser.add_argument('--disable-amp', action='store_true',
    #                     help='disable mixed-precision training (requires more memory and compute)')
    # parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # # System
    # parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    # parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
    #                     help='number of data loading workers per process')
    # parser.add_argument('--evaluate_3d', default=True, type=boolean_string, help='eval 3d only')
    # parser.add_argument('--world-size', default=1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=0, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument('--dist-url', default='env://', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str)
    # parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    # parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    # parser.add_argument('--test_ckpt_addr', default='', help='the ckpt to test 3d zero shot')
    
    
    # others
    parser.add_argument(
        '--outfile', help='output file when running by script instead of pycharm', type=str)
    parser.add_argument('--hostname', help='hostname of the server', type=str)
    parser.add_argument('--remark', help='comments on this experiment',
                        type=str, default='decrease radius by 0.5mm to 5 mm')
    parser.add_argument('--jobid', help='slurm job_id', type=int, default=0)
    # For jupyter notebooks
    if jupyter:
        parser.add_argument(
            "--f", help="a dummy argument to fool ipython", default="0")

        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    if args.x_size == 0 or args.y_size == 0:
        raise Exception("0 x_size or y_size: ")
    if args.net == 'mlp_reg':
        args.set_all_xyz_to_1 = True
    if args.input_mode == 'vessel_skeleton_pcd':
        args.ct_sp = 'ori'
        args.batch_size = 5
    elif args.input_mode == 'lung_mask_pcd':
        args.ct_sp = '1.5'
        args.PNB = 28000
        args.batch_size = 5
    elif args.input_mode == 'modelnet40_pcd':
        args.target = '-'*39  # it.split('-') =40
        args.loss = 'ce'
        args.total_folds = 1
        args.batch_size = 20
    else:
        pass
    return args


if __name__ == "__main__":
    get_args()
